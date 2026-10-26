// k6 load test for the authenticated CouncilAI API
// (reference: tests/load/load_test.js).
//
// Differences from the reference, all deliberate:
//  * No hard-coded account or password: setup() registers fresh users with
//    random credentials and uses the tokens from real responses. Nothing
//    secret is written to the summary.
//  * No hard-coded document: setup() uploads tests/load/corpus.md through the
//    authenticated ingestion endpoint for every user (each user gets their
//    own grant; identical bytes share one content-addressed doc_id).
//  * Every query carries the document, so no profile triggers live web
//    search (no load on third-party sites).
//  * Rate limiting stays on. Profiles stay within the per-user quota by
//    spreading VUs over several users; HTTP 429 is counted separately and
//    is not reported as application failure. The "ratelimit" profile
//    exceeds the quota on purpose.
//  * Cache hits are classified by tier (l1_exact / l2_semantic); the
//    reference called every hit "semantic".
//  * The summary never crashes on missing metrics.
//
// Usage (from the repository root, stack running):
//   docker run --rm --network councilai-reproduced_default -v "$PWD/tests/load:/scripts" \
//     -e PROFILE=reference grafana/k6:0.54.0 run /scripts/load_test.js
//
// PROFILE: reference | cold | l1 | l2 | direct | council | deep | multiuser | ratelimit

import http from 'k6/http';
import { check, sleep, fail } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';
import { randomBytes } from 'k6/crypto';
import exec from 'k6/execution';

const BASE = (__ENV.API_BASE || 'http://go-backend:8080').replace(/\/$/, '');
const PROFILE = __ENV.PROFILE || 'reference';
const CORPUS = open('./corpus.md');

// ── Custom metrics ──────────────────────────────────────────────────
const tierNone = new Trend('latency_cold_ms', true);
const tierL1 = new Trend('latency_l1_hit_ms', true);
const tierL2 = new Trend('latency_l2_hit_ms', true);
const cacheHit = new Rate('cache_hit_rate');
const l1Hit = new Rate('l1_hit_rate');
const l2Hit = new Rate('l2_hit_rate');
const appFailed = new Rate('app_failure_rate'); // non-2xx except 429
const status = {
  s200: new Counter('status_200'), s401: new Counter('status_401'), s403: new Counter('status_403'),
  s404: new Counter('status_404'), s429: new Counter('status_429'), s503: new Counter('status_503'),
  s5xx: new Counter('status_5xx_other'), other: new Counter('status_other'),
};
const crossUserHits = new Counter('cross_user_first_ask_hits');

// Questions answerable from the corpus.
const QUESTIONS = [
  'Who founded the Mirevale Aqueduct Authority?',
  'How is water treated before distribution?',
  'What is the average daily delivery of the aqueduct?',
  'How often are old channel sections inspected?',
  'What do households pay for water?',
];
// Paraphrases of QUESTIONS (same order) for the L2 profile.
const PARAPHRASES = [
  'Who established the Mirevale Aqueduct Authority?',
  'What treatment does the water receive before it is distributed?',
  'How much water does the aqueduct deliver on an average day?',
  'How frequently are the oldest channel sections inspected?',
  'How much are households charged for water?',
];

// ── Profiles ────────────────────────────────────────────────────────
const quiet = { http_req_failed: ['rate<1'] }; // http_req_failed is informational: 429s count there
const PROFILES = {
  // Reference staged structure: 0→5 VUs (10 s), →20 (30 s), →0 (10 s).
  reference: {
    users: 10,
    options: { stages: [{ duration: '10s', target: 5 }, { duration: '30s', target: 20 }, { duration: '10s', target: 0 }] },
    thresholds: { app_failure_rate: ['rate<0.01'], http_req_duration: ['p(95)<2000'], cache_hit_rate: ['rate>0.70'], latency_l1_hit_ms: ['p(95)<50'] },
    pick: () => ({ question: QUESTIONS[rnd(QUESTIONS.length)] }),
    pause: 0.5,
  },
  cold: { users: 5, options: { vus: 5, duration: '40s' }, thresholds: { app_failure_rate: ['rate<0.01'] },
    pick: () => ({ question: `${QUESTIONS[rnd(QUESTIONS.length)]} (run ${uid()})` }), pause: 0.5 },
  l1: { users: 10, warm: true, options: { vus: 20, duration: '40s' },
    thresholds: { app_failure_rate: ['rate<0.01'], l1_hit_rate: ['rate>0.99'], latency_l1_hit_ms: ['p(95)<50'] },
    pick: () => ({ question: QUESTIONS[rnd(QUESTIONS.length)] }), pause: 0.5 },
  l2: { users: 5, warm: true, needsL2: true, options: { vus: 5, iterations: 25 },
    thresholds: { app_failure_rate: ['rate<0.01'] },
    pick: (it) => ({ question: `${PARAPHRASES[it % PARAPHRASES.length]}` }), pause: 0.5 },
  direct: { users: 5, options: { vus: 5, duration: '40s' }, thresholds: { app_failure_rate: ['rate<0.01'] },
    pick: () => ({ question: `${QUESTIONS[rnd(QUESTIONS.length)]} (run ${uid()})`, strategy: 'direct' }), pause: 0.5 },
  council: { users: 5, options: { vus: 5, duration: '40s' }, thresholds: { app_failure_rate: ['rate<0.01'] },
    pick: () => ({ question: `${QUESTIONS[rnd(QUESTIONS.length)]} (run ${uid()})`, strategy: 'council' }), pause: 0.5 },
  deep: { users: 5, options: { vus: 5, duration: '40s' }, thresholds: { app_failure_rate: ['rate<0.01'] },
    pick: () => ({ question: `${QUESTIONS[rnd(QUESTIONS.length)]} (run ${uid()})`, strategy: 'council_deep' }), pause: 0.5 },
  // One user per VU; every user asks the shared question set. A cache hit on
  // a user's first ask of a question would mean cross-user reuse.
  multiuser: { users: 10, perVU: true, options: { vus: 10, duration: '40s' },
    thresholds: { app_failure_rate: ['rate<0.01'], cross_user_first_ask_hits: ['count==0'] },
    pick: () => ({ question: QUESTIONS[rnd(QUESTIONS.length)] }), pause: 0.5 },
  // Deliberately above one user's quota (10 req/s, burst 20): expect 429s.
  ratelimit: { users: 1, options: { vus: 20, duration: '20s' }, thresholds: { app_failure_rate: ['rate<0.01'], status_429: ['count>0'] },
    pick: () => ({ question: QUESTIONS[rnd(QUESTIONS.length)] }), pause: 0 },
};
const P = PROFILES[PROFILE];
if (!P) throw new Error(`unknown PROFILE ${PROFILE}`);

export const options = Object.assign({}, P.options, {
  setupTimeout: '300s',
  thresholds: Object.assign({}, quiet, P.thresholds),
  summaryTrendStats: ['avg', 'min', 'med', 'p(90)', 'p(95)', 'p(99)', 'max', 'count'],
});

function rnd(n) { return Math.floor(Math.random() * n); }
function uid() { return hex(6); }
function hex(n) { return Array.from(new Uint8Array(randomBytes(n))).map((b) => b.toString(16).padStart(2, '0')).join(''); }
const JSONH = { 'Content-Type': 'application/json' };

// POST with Retry-After handling, for setup only.
function postPatiently(path, body, params) {
  for (let i = 0; i < 40; i++) {
    const r = http.post(`${BASE}${path}`, body, params);
    if (r.status !== 429) return r;
    sleep(Number(r.headers['Retry-After'] || 5));
  }
  fail(`${path}: still rate limited`);
}

// ── Setup: users, document, optional cache warm-up ──────────────────
export function setup() {
  const users = [];
  for (let i = 0; i < P.users; i++) {
    const username = `k6${hex(5)}`;
    const r = postPatiently('/api/v1/register', JSON.stringify({ username, password: hex(16) }), { headers: JSONH });
    if (r.status !== 201) fail(`register: ${r.status} ${r.body}`);
    users.push({ token: r.json('token') });
  }
  let docId = '';
  for (const u of users) {
    const r = postPatiently('/api/v1/ingest', { file: http.file(CORPUS, 'mirevale_handbook.md', 'text/markdown') },
      { headers: { Authorization: `Bearer ${u.token}` }, timeout: '120s' });
    if (r.status !== 200) fail(`ingest: ${r.status} ${r.body}`);
    docId = r.json('doc_id');
  }
  // L2 needs explicit, benchmark-only enablement on the server.
  if (P.needsL2) {
    const r = postPatiently('/api/v1/query', JSON.stringify({ question: `l2 probe ${hex(4)}`, doc_id: docId, strategy: 'council' }),
      { headers: Object.assign({ Authorization: `Bearer ${users[0].token}` }, JSONH), timeout: '120s' });
    if (r.json('cache.l2') === 'disabled_by_policy') fail('L2 reuse is disabled; start go-backend with SEMANTIC_CACHE_ENABLED=true for PROFILE=l2');
  }
  if (P.warm) {
    for (const u of users) {
      for (const q of QUESTIONS) {
        const r = postPatiently('/api/v1/query', JSON.stringify({ question: q, doc_id: docId }),
          { headers: Object.assign({ Authorization: `Bearer ${u.token}` }, JSONH), timeout: '120s' });
        if (r.status !== 200) fail(`warm-up: ${r.status}`);
      }
    }
  }
  return { users, docId, started: new Date().toISOString() };
}

// Per-VU memory of questions already asked (multiuser profile).
const asked = {};

export default function (data) {
  const vu = exec.vu.idInTest;
  const user = P.perVU ? data.users[(vu - 1) % data.users.length] : data.users[vu % data.users.length];
  const body = Object.assign({ doc_id: data.docId, top_k: 4 }, P.pick(exec.scenario.iterationInTest));
  const res = http.post(`${BASE}/api/v1/query`, JSON.stringify(body), {
    headers: Object.assign({ Authorization: `Bearer ${user.token}` }, JSONH), timeout: '150s',
    tags: { profile: PROFILE },
  });
  record(res, body.question);
  if (P.pause) sleep(P.pause);
}

function record(res, question) {
  const s = res.status;
  if (s === 200) status.s200.add(1);
  else if (s === 401) status.s401.add(1);
  else if (s === 403) status.s403.add(1);
  else if (s === 404) status.s404.add(1);
  else if (s === 429) status.s429.add(1);
  else if (s === 503) status.s503.add(1);
  else if (s >= 500) status.s5xx.add(1);
  else status.other.add(1);
  if (s === 429) return; // quota response: neither success nor application failure
  appFailed.add(s !== 200);
  if (s !== 200) return;
  const ok = check(res, { 'has answer': (r) => typeof r.json('answer') === 'string' && r.json('answer').length > 0 });
  if (!ok) return;
  const tier = res.json('cache.tier') || 'none';
  const ms = res.timings.duration;
  cacheHit.add(tier !== 'none');
  l1Hit.add(tier === 'l1_exact');
  if (tier === 'l1_exact') tierL1.add(ms);
  else if (tier === 'l2_semantic') { tierL2.add(ms); l2Hit.add(true); }
  else { tierNone.add(ms); if (PROFILE === 'l2') l2Hit.add(false); }
  if (P.perVU) {
    const key = `${exec.vu.idInTest}|${question}`;
    if (!asked[key] && tier !== 'none') crossUserHits.add(1);
    asked[key] = true;
  }
}

// ── Summary (robust to missing metrics) ─────────────────────────────
function val(data, name, stat) {
  const m = data.metrics[name];
  if (!m || !m.values || m.values[stat] === undefined) return null;
  return m.values[stat];
}
function fmt(v, unit = '', digits = 1) { return v === null ? 'n/a' : `${Number(v).toFixed(digits)}${unit}`; }

export function handleSummary(data) {
  const stamp = new Date().toISOString().replace(/[:.]/g, '-');
  const lat = (name) => `p50 ${fmt(val(data, name, 'med'), ' ms')} | p95 ${fmt(val(data, name, 'p(95)'), ' ms')} | p99 ${fmt(val(data, name, 'p(99)'), ' ms')} | n=${fmt(val(data, name, 'count'), '', 0)}`;
  const thresholds = Object.entries(data.metrics)
    .filter(([, m]) => m.thresholds)
    .map(([n, m]) => `  ${Object.values(m.thresholds).every((t) => t.ok) ? 'PASS' : 'FAIL'} ${n}: ${Object.keys(m.thresholds).join(', ')}`)
    .join('\n');
  const text = `
k6 CouncilAI load test — profile "${PROFILE}"
  requests        ${fmt(val(data, 'http_reqs', 'count'), '', 0)} incl. setup (${fmt(val(data, 'http_reqs', 'rate'), ' req/s', 2)}), measured queries ${fmt(val(data, 'iterations', 'count'), '', 0)}
  max VUs         ${fmt(val(data, 'vus_max', 'value'), '', 0)}
  all requests    ${lat('http_req_duration')}
  cold (models)   ${lat('latency_cold_ms')}
  L1 hits         ${lat('latency_l1_hit_ms')}
  L2 hits         ${lat('latency_l2_hit_ms')}
  cache hit rate  ${fmt(val(data, 'cache_hit_rate', 'rate') === null ? null : val(data, 'cache_hit_rate', 'rate') * 100, '%')} (L1 ${fmt(val(data, 'l1_hit_rate', 'rate') === null ? null : val(data, 'l1_hit_rate', 'rate') * 100, '%')}, L2 ${fmt(val(data, 'l2_hit_rate', 'rate') === null ? null : val(data, 'l2_hit_rate', 'rate') * 100, '%')})
  app failures    ${fmt(val(data, 'app_failure_rate', 'rate') === null ? null : val(data, 'app_failure_rate', 'rate') * 100, '%', 2)} (non-2xx excluding 429)
  statuses        200=${fmt(val(data, 'status_200', 'count'), '', 0)} 401=${fmt(val(data, 'status_401', 'count'), '', 0)} 403=${fmt(val(data, 'status_403', 'count'), '', 0)} 404=${fmt(val(data, 'status_404', 'count'), '', 0)} 429=${fmt(val(data, 'status_429', 'count'), '', 0)} 503=${fmt(val(data, 'status_503', 'count'), '', 0)} other5xx=${fmt(val(data, 'status_5xx_other', 'count'), '', 0)}
  cross-user hits ${fmt(val(data, 'cross_user_first_ask_hits', 'count'), '', 0)}
thresholds
${thresholds || '  (none)'}
`;
  // setup_data holds the test users' JWTs: never written to disk.
  const raw = Object.assign({}, data);
  delete raw.setup_data;
  return {
    stdout: text,
    [`/scripts/results/${PROFILE}-${stamp}.json`]: JSON.stringify(raw, null, 2),
    [`/scripts/results/${PROFILE}-${stamp}.txt`]: text,
  };
}

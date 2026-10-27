#!/usr/bin/env python3
"""Provision, verify and clean up the QuorumAI Phi-4-mini vLLM endpoint on
RunPod Serverless.

Every setting comes from the `endpoint` section of
evaluation/config/experiments/runpod_phi4_bench.yaml, the same file that
`gpu_bench --check-endpoint` verifies against.

Tools:
- `runpodctl` (official CLI, >= 3.0) for everything it can express: account
  balance, GPU catalog, template create/list/delete, endpoint
  list/update/health/delete.
- RunPod REST v1 `POST /v1/endpoints` for endpoint creation only. The CLI
  accepts a single GPU pool and a *minimum* CUDA version; this endpoint needs an
  exact GPU-type list (the AMPERE_24 pool also holds an RTX PRO 6000 MIG slice)
  and `allowedCudaVersions: ["13.0"]`.
- REST v1 `GET /v1/endpoints/{id}` to verify the live settings.

Commands:
  plan       offline: print the exact template command, endpoint payload, costs
  preflight  read-only: CLI version, key, balance, GPU availability, image
             digest, model revision, existing resources (what deploy would do)
  deploy     --authorize-deploy: create (or reuse) the template and endpoint.
             Idempotent: resources are found by exact name, so re-runs never
             duplicate them; a same-named resource with different settings is
             a hard stop, never silently changed. Sends no inference.
  update     --authorize-update: bring an existing endpoint to the config with
             PATCH /v1/endpoints/{id}. Sends every approved setting (the API
             does not document whether omitted fields keep their values),
             refuses while workers or jobs are active, then re-verifies.
  status     read-only: live settings check and worker/job counts
  cleanup    --yes: scale to 0 workers, wait for zero, delete endpoint, then
             template. Idempotent: nothing left is a success.

Credentials: RUNPOD_MANAGEMENT_API_KEY from the environment only (an "All"
key: creating templates and endpoints needs write access). It is passed to
runpodctl as that process's RUNPOD_API_KEY, in a minimal environment, so the
benchmark's restricted inference key (also called RUNPOD_API_KEY in .env)
never reaches the CLI. Keys are never printed, logged or written.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))
from councileval.gpu_bench import check_settings, serverless_placement  # noqa: E402  (same checks as the benchmark)
from councileval.gpu_budget import campaign_plan, load_campaign  # noqa: E402

DEFAULT_CONFIG = ROOT / "evaluation/config/experiments/runpod_phi4_bench.yaml"
DEFAULT_CAMPAIGN = ROOT / "evaluation/config/experiments/runpod_campaign.yaml"
STATE_DIR = ROOT / "deploy/runpod/state"          # git-ignored; holds ids, never keys
KEY_ENV = "RUNPOD_MANAGEMENT_API_KEY"
REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
MIN_CLI = (3, 0, 0)
USER_AGENT = "quorumai-deploy/1.0"
_SECRET = re.compile(r"(rpa_|rps_)[A-Za-z0-9]+")


class Stop(RuntimeError):
    """A check failed: nothing (more) is changed."""


def redact(text: str, key: str = "") -> str:
    text = _SECRET.sub("<redacted>", text or "")
    return text.replace(key, "<redacted>") if key else text


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── configuration ────────────────────────────────────────────────────

def load(path: Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    e = cfg["endpoint"]
    need = ("name", "template_name", "endpoint_type", "compute_type", "image", "image_digest", "container_disk_gb",
            "network_volume", "data_center_ids", "gpu_count", "max_workers", "min_workers", "idle_timeout_s",
            "execution_timeout_s", "flashboot", "scaler_type", "scaler_value", "allowed_cuda_versions", "gpu_type_ids", "env")
    missing = [k for k in need if k not in e]
    if missing:
        raise Stop(f"config endpoint section is missing {missing}: every setting must be explicit")
    if e["endpoint_type"] != "queue" or e["compute_type"] != "GPU":
        raise Stop("only a queue-based GPU endpoint is supported")
    if e["network_volume"] not in (None, "none") or e["data_center_ids"] != "any":
        raise Stop("network volumes and data-centre pinning are not part of the approved deployment")
    if e["min_workers"] != 0 or e["max_workers"] != 1 or e["gpu_count"] != 1:
        raise Stop("approved deployment is min 0 / max 1 workers, one GPU each")
    if ":" not in e["image"] or e["image"].endswith(":latest") or not e["image_digest"].startswith("sha256:"):
        raise Stop("image must be pinned by tag and digest")
    if e["env"].get("MODEL_REVISION") in (None, "", "main"):
        raise Stop("MODEL_REVISION must be a pinned commit")
    return cfg


def template_env(cfg: dict) -> dict[str, str]:
    return {k: str(v) for k, v in cfg["endpoint"]["env"].items()}


def template_args(cfg: dict) -> list[str]:
    e = cfg["endpoint"]
    return ["template", "create", "--name", e["template_name"], "--image", e["image"],
            "--container-disk-in-gb", str(e["container_disk_gb"]), "--serverless",
            "--env", json.dumps(template_env(cfg), separators=(",", ":")), "-o", "json"]


def endpoint_payload(cfg: dict, template_id: str) -> dict:
    """REST v1 POST /endpoints body. No dataCenterIds (any) and no network volume."""
    e = cfg["endpoint"]
    return {"name": e["name"], "templateId": template_id, "computeType": "GPU",
            "gpuTypeIds": list(e["gpu_type_ids"]), "gpuCount": e["gpu_count"],
            "allowedCudaVersions": list(e["allowed_cuda_versions"]),
            "workersMin": e["min_workers"], "workersMax": e["max_workers"],
            "idleTimeout": e["idle_timeout_s"], "executionTimeoutMs": e["execution_timeout_s"] * 1000,
            "flashboot": bool(e["flashboot"]), "scalerType": e["scaler_type"], "scalerValue": e["scaler_value"]}


# ── provider access ──────────────────────────────────────────────────

class Provider:
    """runpodctl + REST v1 with the management key. Every call is listed in
    `calls` (method and target only) for the record."""

    def __init__(self, key: str, cli: str = "runpodctl", rest: str = REST, opener=urllib.request.urlopen, graphql: str = GRAPHQL):
        if not key:
            raise Stop(f"{KEY_ENV} is not set (read from the environment only)")
        self._key, self.cli, self.rest, self.open, self.graphql = key, cli, rest.rstrip("/"), opener, graphql
        self.calls: list[str] = []

    def ctl(self, *args: str, timeout: float = 60) -> object:
        exe = shutil.which(self.cli) or self.cli
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/tmp"),
               "RUNPOD_API_KEY": self._key}
        shown = [a if not a.startswith("{") else "<env json>" for a in args]
        self.calls.append("runpodctl " + " ".join(shown))
        p = subprocess.run([exe, *args], env=env, capture_output=True, text=True, timeout=timeout)
        out = redact(p.stdout, self._key)
        if p.returncode != 0:
            raise Stop(f"runpodctl {' '.join(shown[:3])} failed: {redact(p.stderr or p.stdout, self._key)[:400]}")
        try:
            data = json.loads(out) if out.strip() else None
        except ValueError:
            return out
        if isinstance(data, dict) and data.get("error"):
            raise Stop(f"runpodctl {' '.join(shown[:3])}: {data.get('code')}: {str(data['error'])[:300]}")
        return data

    def http(self, method: str, url: str, body: Optional[dict] = None, auth: bool = True, timeout: float = 30):
        self.calls.append(f"{method} {url.split('?')[0]}")
        # An explicit User-Agent: RunPod's edge (Cloudflare) rejects Python's
        # default "Python-urllib/x.y" with 403 "error code: 1010".
        h = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if auth:
            h["Authorization"] = f"Bearer {self._key}"
        req = urllib.request.Request(url, method=method, headers=h, data=json.dumps(body).encode() if body is not None else None)
        try:
            with self.open(req, timeout=timeout) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as exc:
            detail = redact(exc.read().decode("utf-8", "replace"), self._key)[:400]
            return exc.code, {"error": detail}

    def gql(self, query: str, variables: Optional[dict] = None) -> dict:
        code, body = self.http("POST", self.graphql, {"query": query, "variables": variables or {}})
        if code != 200 or (body or {}).get("errors"):
            raise Stop(f"GraphQL HTTP {code}: {redact(json.dumps((body or {}).get('errors') or body), self._key)[:400]}")
        return body["data"]

    def version(self) -> tuple[int, ...]:
        exe = shutil.which(self.cli) or self.cli
        p = subprocess.run([exe, "version"], capture_output=True, text=True, timeout=20,
                           env={"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/tmp")})
        out = p.stdout + p.stderr
        m = re.search(r"(\d+)\.(\d+)\.(\d+)", out)
        if not m:
            raise Stop(f"cannot read runpodctl version from {out[:80]!r}")
        return tuple(int(x) for x in m.groups())

    # lookups by exact name (the idempotency key)
    def templates_named(self, name: str) -> list[dict]:
        return [t for t in (self.ctl("template", "list", "--type", "user", "-o", "json") or []) if t.get("name") == name]

    def endpoints_named(self, name: str) -> list[dict]:
        return [e for e in (self.ctl("serverless", "list", "-o", "json") or []) if e.get("name") == name]

    def endpoint_settings(self, endpoint_id: str) -> dict:
        code, body = self.http("GET", f"{self.rest}/endpoints/{endpoint_id}?includeTemplate=true&includeWorkers=true")
        if code != 200:
            raise Stop(f"reading endpoint {endpoint_id}: HTTP {code} {body}")
        t = body.get("template") or {}
        if body.get("templateId") and not t.get("imageName"):
            # The endpoint's embedded template omits imageName; the template route has it.
            code, tpl = self.http("GET", f"{self.rest}/templates/{body['templateId']}")
            if code != 200:
                raise Stop(f"reading template {body['templateId']}: HTTP {code} {tpl}")
            body["template"] = {**t, **tpl}
        return body

    def health(self, endpoint_id: str) -> dict:
        return self.ctl("serverless", "health", endpoint_id, "-o", "json") or {}


# ── public checks (no key) ───────────────────────────────────────────

def registry_digest(image: str, opener=urllib.request.urlopen) -> Optional[str]:
    repo, tag = image.rsplit(":", 1)
    tok = json.load(opener(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull", timeout=20))["token"]
    acc = ("application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, "
           "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json")
    req = urllib.request.Request(f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}", method="HEAD",
                                 headers={"Authorization": f"Bearer {tok}", "Accept": acc})
    try:
        with opener(req, timeout=20) as r:
            return r.headers.get("Docker-Content-Digest")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def hf_revision(model: str, revision: str, opener=urllib.request.urlopen) -> dict:
    try:
        d = json.load(opener(f"https://huggingface.co/api/models/{model}/revision/{revision}", timeout=20))
    except urllib.error.HTTPError as exc:
        return {"exists": False, "status": exc.code}
    return {"exists": d.get("sha") == revision, "gated": d.get("gated"),
            "architectures": (d.get("config") or {}).get("architectures")}


# ── commands ─────────────────────────────────────────────────────────

def plan(cfg: dict, campaign: Optional[dict]) -> dict:
    e = cfg["endpoint"]
    cmd = "runpodctl " + " ".join(a if not a.startswith("{") else f"'{a}'" for a in template_args(cfg))
    out = {"template_command": cmd,
           "endpoint_request": {"method": "POST", "url": f"{REST}/endpoints",
                                "body": endpoint_payload(cfg, "<template id from the command above>")},
           "not_set_on_purpose": {"dataCenterIds": "any data centre", "networkVolumeId": "none",
                                  "workersMin": "0: nothing runs or bills while idle"},
           "deployment_cost_usd": 0.0,
           "deployment_cost_note": "creating a template and a min-0 endpoint starts no worker; image pulls are not billed. "
                                   "GPU time is billed only once a request starts a worker."}
    if campaign:
        out["benchmark_costs"] = campaign_plan(campaign)
        out["benchmark_costs"]["price_usd_per_hour"] = campaign["price_usd_per_hour"]
    out["cleanup"] = [f"runpodctl serverless update <id> --workers-min 0 --workers-max 0",
                      "poll runpodctl serverless health <id> until no workers",
                      "runpodctl serverless delete <id>", f"runpodctl template delete <template id of {e['template_name']}>"]
    return out


def preflight(cfg: dict, pv: Provider, campaign: Optional[dict], digest_fn: Callable = registry_digest,
              hf_fn: Callable = hf_revision, placement_fn: Callable = serverless_placement) -> dict:
    """Read-only. Returns findings and the action deploy would take."""
    e, problems, rep = cfg["endpoint"], [], {"utc": now()}
    v = pv.version()
    rep["runpodctl_version"] = ".".join(map(str, v))
    if v < MIN_CLI:
        problems.append(f"runpodctl {rep['runpodctl_version']} < {'.'.join(map(str, MIN_CLI))}")
    user = pv.ctl("user", "-o", "json") or {}
    rep["account"] = {k: user.get(k) for k in ("clientBalance", "currentSpendPerHr", "spendLimit")}
    need = (campaign or {}).get("max_gpu_spend_usd", 0)
    if (user.get("clientBalance") or 0) < need:
        problems.append(f"balance ${user.get('clientBalance')} is below the approved GPU cap ${need}")
    gpus = {g.get("gpuId"): g for g in (pv.ctl("gpu", "list", "--include-unavailable", "-o", "json") or [])}
    rep["gpus"] = {}
    for gid in e["gpu_type_ids"]:
        g = gpus.get(gid)
        if not g:
            problems.append(f"GPU type {gid!r} is not in RunPod's catalog")
            continue
        rep["gpus"][gid] = {"memoryInGb": g.get("memoryInGb"), "available": g.get("available"), "stockStatus": g.get("stockStatus")}
        if g.get("memoryInGb") != 24:
            problems.append(f"{gid} has {g.get('memoryInGb')} GB, expected 24")
    if not any(x.get("available") for x in rep["gpus"].values()):
        problems.append("none of the approved GPU types is available right now")
    # Serverless placement: an available host must match GPU type AND an allowed CUDA version (exact set).
    pl = placement_fn(cfg, pv._key)
    rep["serverless_placement"] = pl
    if not pl.get("ok"):
        problems.append(f"no Serverless host matches GPU {e['gpu_type_ids']} with CUDA {e['allowed_cuda_versions']}")
    got = digest_fn(e["image"])
    rep["image"] = {"image": e["image"], "registry_digest": got, "pinned_digest": e["image_digest"]}
    if got is None:
        problems.append(f"image {e['image']} does not exist in the registry")
    elif got != e["image_digest"]:
        problems.append(f"image tag {e['image']} now points to {got}, not the pinned {e['image_digest']}")
    hf = hf_fn(e["env"]["MODEL_NAME"], e["env"]["MODEL_REVISION"])
    rep["model"] = {"model": e["env"]["MODEL_NAME"], "revision": e["env"]["MODEL_REVISION"], **hf}
    if not hf.get("exists"):
        problems.append("pinned model revision does not resolve on Hugging Face")
    elif hf.get("gated"):
        problems.append("model is gated: the worker would need an HF token")
    tpls, eps = pv.templates_named(e["template_name"]), pv.endpoints_named(e["name"])
    rep["existing"] = {"templates": [t.get("id") for t in tpls], "endpoints": [x.get("id") for x in eps]}
    if len(tpls) > 1 or len(eps) > 1:
        problems.append("more than one resource with the configured name: clean up before deploying")
    rep["deploy_would"] = {"template": "reuse (after verification)" if tpls else "create",
                           "endpoint": "reuse (after verification)" if eps else "create"}
    rep["permissions"] = ("read access confirmed (account, catalog, templates, endpoints). Write access is first "
                          "exercised by template creation, which is free; deploy stops there if it is refused.")
    rep["problems"] = problems
    return rep


def _template_problems(t: dict, cfg: dict) -> list[str]:
    e, p = cfg["endpoint"], []
    if t.get("imageName") != e["image"]:
        p.append(f"template image {t.get('imageName')}, expected {e['image']}")
    if t.get("isServerless") is not True:
        p.append("template is not a serverless template")
    if t.get("containerDiskInGb") != e["container_disk_gb"]:
        p.append(f"template disk {t.get('containerDiskInGb')} GB, expected {e['container_disk_gb']}")
    if t.get("volumeInGb"):
        p.append("template has a volume")
    if {k: str(v) for k, v in (t.get("env") or {}).items()} != template_env(cfg):
        p.append("template env differs from the config")
    return p


def deploy(cfg: dict, pv: Provider, campaign: Optional[dict], authorize: bool, state_dir: Path = STATE_DIR, **pre) -> dict:
    if not authorize:
        raise Stop("refusing: deploy needs --authorize-deploy")
    rep = preflight(cfg, pv, campaign, **pre)
    if rep["problems"]:
        raise Stop("preflight failed: " + "; ".join(rep["problems"]))
    e = cfg["endpoint"]
    # Template: reuse only if it matches exactly.
    tpls = pv.templates_named(e["template_name"])
    if tpls:
        t = tpls[0]
        full = pv.ctl("template", "get", t["id"], "-o", "json") or t
        tp = _template_problems(full, cfg)
        if tp:
            raise Stop(f"existing template {t['id']} differs ({'; '.join(tp)}); run cleanup first")
        template_id, template_action = t["id"], "reused"
    else:
        created = pv.ctl(*template_args(cfg))
        if not isinstance(created, dict) or not created.get("id"):
            raise Stop(f"template create returned no id: {str(created)[:200]}")
        template_id, template_action = created["id"], "created"
    # Endpoint: reuse only if the live settings pass the same check as --check-endpoint.
    eps = pv.endpoints_named(e["name"])
    if eps:
        endpoint_id, endpoint_action = eps[0]["id"], "reused"
    else:
        code, body = pv.http("POST", f"{pv.rest}/endpoints", endpoint_payload(cfg, template_id))
        if code not in (200, 201) or not (body or {}).get("id"):
            raise Stop(f"endpoint create failed: HTTP {code} {body}; template {template_id} kept (cleanup removes it)")
        endpoint_id, endpoint_action = body["id"], "created"
    live = pv.endpoint_settings(endpoint_id)
    probs = check_settings(live, cfg) + _deploy_only_problems(live, cfg, template_id)
    health = pv.health(endpoint_id)
    state = {"endpoint_id": endpoint_id, "endpoint_name": e["name"], "endpoint_action": endpoint_action,
             "template_id": template_id, "template_name": e["template_name"], "template_action": template_action,
             "image": e["image"], "image_digest": e["image_digest"], "deployed_utc": now(),
             "settings_problems": probs, "health_after_deploy": health, "preflight": rep, "provider_calls": pv.calls}
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"{e['name']}.json").write_text(json.dumps(state, indent=2) + "\n")
    if probs:
        raise Stop(f"endpoint {endpoint_id} settings do not match: {'; '.join(probs)}; nothing was invoked. Run cleanup.")
    return state


def _deploy_only_problems(live: dict, cfg: dict, template_id: str) -> list[str]:
    """Identity checks on top of gpu_bench.check_settings (which covers every setting)."""
    e, p = cfg["endpoint"], []
    if live.get("name") != e["name"]:
        p.append(f"name {live.get('name')!r}, expected {e['name']!r}")
    if live.get("templateId") not in (None, template_id):
        p.append(f"templateId {live.get('templateId')}, expected {template_id}")
    return p


# REST PATCH fields. allowedCudaVersions is not among them: the REST schema
# only accepts values up to "13.0" (HTTP 400 for "13.2", 2026-10-27), so it is
# set through GraphQL saveEndpoint (set_cuda_versions).
PATCHABLE = ("gpuTypeIds", "gpuCount", "workersMin", "workersMax", "idleTimeout",
             "executionTimeoutMs", "flashboot", "scalerType", "scalerValue", "templateId")

_GQL_FIELDS = """id name templateId gpuIds gpuCount instanceIds workersMin workersMax locations networkVolumeId
                 networkVolumeIds { networkVolumeId } idleTimeout scalerType scalerValue executionTimeoutMs
                 minCudaVersion flashBootType allowedCudaVersions modelReferences"""


def set_cuda_versions(pv: Provider, eid: str, versions: list[str]) -> dict:
    """GraphQL saveEndpoint is a full replace: read every field, change only
    allowedCudaVersions (a comma-separated string there), write all of it back,
    and return what the API now reports."""
    cur = (pv.gql("query($id: String!) { myself { endpoint(id: $id) { %s } } }" % _GQL_FIELDS, {"id": eid})
           .get("myself") or {}).get("endpoint")
    if not cur or not cur.get("gpuIds") or not cur.get("templateId"):
        raise Stop(f"GraphQL read of {eid} is incomplete ({cur}); refusing a full-replace write")
    inp = {k: cur.get(k) for k in ("id", "name", "templateId", "gpuIds", "gpuCount", "instanceIds", "workersMin", "workersMax",
                                   "locations", "networkVolumeId", "idleTimeout", "scalerType", "scalerValue",
                                   "executionTimeoutMs", "minCudaVersion")}
    inp["networkVolumeIds"] = [{"networkVolumeId": n["networkVolumeId"]} for n in (cur.get("networkVolumeIds") or [])]
    inp["flashBootType"] = cur.get("flashBootType") or "OFF"
    inp["modelReferences"] = cur.get("modelReferences") or []
    inp["allowedCudaVersions"] = ",".join(versions)
    saved = pv.gql("mutation($input: EndpointInput!) { saveEndpoint(input: $input) { %s } }" % _GQL_FIELDS, {"input": inp})["saveEndpoint"]
    unchanged = {k: (cur.get(k), saved.get(k)) for k in cur if k != "allowedCudaVersions" and cur.get(k) != saved.get(k)}
    if unchanged:
        raise Stop(f"saveEndpoint changed other fields: {unchanged}")
    return {"before": cur.get("allowedCudaVersions"), "after": saved.get("allowedCudaVersions")}


def update(cfg: dict, pv: Provider, authorize: bool, state_dir: Path = STATE_DIR) -> dict:
    """Reconcile the existing endpoint with the config (no create, no delete)."""
    if not authorize:
        raise Stop("refusing: update needs --authorize-update")
    e = cfg["endpoint"]
    eps = pv.endpoints_named(e["name"])
    if len(eps) != 1:
        raise Stop(f"expected exactly one endpoint named {e['name']!r}, found {len(eps)}")
    eid = eps[0]["id"]
    tpls = pv.templates_named(e["template_name"])
    if len(tpls) != 1:
        raise Stop(f"expected exactly one template named {e['template_name']!r}, found {len(tpls)}")
    tp = _template_problems(pv.ctl("template", "get", tpls[0]["id"], "-o", "json") or tpls[0], cfg)
    if tp:
        raise Stop(f"template differs from the config ({'; '.join(tp)}); not updating")
    h = pv.health(eid)
    jobs = (h.get("jobs") or {})
    if workers_up(h) or jobs.get("inQueue") or jobs.get("inProgress"):
        raise Stop(f"endpoint {eid} has workers or jobs ({h}); update only an idle endpoint")
    before = pv.endpoint_settings(eid)
    desired = endpoint_payload(cfg, tpls[0]["id"])
    body = {k: desired[k] for k in PATCHABLE}
    changed = {k: {"from": before.get(k), "to": body[k]} for k in PATCHABLE if before.get(k) != body[k]}
    if changed:
        code, resp = pv.http("PATCH", f"{pv.rest}/endpoints/{eid}", body)
        if code != 200:
            raise Stop(f"PATCH failed: HTTP {code} {resp}")
    want_cuda = [str(v) for v in e["allowed_cuda_versions"]]
    if sorted(map(str, before.get("allowedCudaVersions") or [])) != sorted(want_cuda):
        changed["allowedCudaVersions"] = {"from": before.get("allowedCudaVersions"), "to": want_cuda,
                                          "graphql": set_cuda_versions(pv, eid, want_cuda)}
    live = pv.endpoint_settings(eid)
    probs = check_settings(live, cfg) + _deploy_only_problems(live, cfg, tpls[0]["id"])
    # check_settings accepts a subset of GPU types; an update must land exactly.
    if sorted(live.get("gpuTypeIds") or []) != sorted(e["gpu_type_ids"]):
        probs.append(f"gpuTypeIds {live.get('gpuTypeIds')} != {e['gpu_type_ids']}")
    if sorted(map(str, live.get("allowedCudaVersions") or [])) != sorted(want_cuda):
        probs.append(f"allowedCudaVersions {live.get('allowedCudaVersions')} != {want_cuda}")
    out = {"endpoint_id": eid, "changed": changed, "settings_problems": probs, "health": pv.health(eid),
           "updated_utc": now(), "provider_calls": pv.calls}
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"{e['name']}.update-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json").write_text(json.dumps(out, indent=2) + "\n")
    if probs:
        raise Stop(f"endpoint {eid} does not match after update: {'; '.join(probs)}")
    return out


def workers_up(h: dict) -> int:
    w = (h or {}).get("workers") or {}
    return sum(int(w.get(k) or 0) for k in ("idle", "running", "initializing", "ready", "throttled"))


def cleanup(cfg: dict, pv: Provider, yes: bool, wait_s: float = 180, sleep: Callable = time.sleep,
            state_dir: Path = STATE_DIR) -> dict:
    if not yes:
        raise Stop("refusing: cleanup deletes the endpoint and template; pass --yes")
    e, done = cfg["endpoint"], {"endpoints_deleted": [], "templates_deleted": [], "utc": now()}
    ids = {x["id"] for x in pv.endpoints_named(e["name"])}
    sf = state_dir / f"{e['name']}.json"
    if sf.exists():  # an id recorded by deploy but renamed since is still ours
        sid = json.loads(sf.read_text()).get("endpoint_id")
        if sid and any(x.get("id") == sid for x in (pv.ctl("serverless", "list", "-o", "json") or [])):
            ids.add(sid)
    for eid in sorted(ids):
        pv.ctl("serverless", "update", eid, "--workers-min", "0", "--workers-max", "0", "-o", "json")
        deadline, zero = time.monotonic() + wait_s, False
        while time.monotonic() < deadline:
            if workers_up(pv.health(eid)) == 0:
                zero = True
                break
            sleep(5)
        if not zero:
            raise Stop(f"endpoint {eid} still reports workers after {wait_s:.0f}s at max 0; not deleting yet, re-run cleanup")
        pv.ctl("serverless", "delete", eid, "-o", "json")
        done["endpoints_deleted"].append(eid)
    for t in pv.templates_named(e["template_name"]):
        pv.ctl("template", "delete", t["id"], "-o", "json")
        done["templates_deleted"].append(t["id"])
    left = {"endpoints": [x.get("id") for x in pv.endpoints_named(e["name"])],
            "templates": [x.get("id") for x in pv.templates_named(e["template_name"])]}
    done["remaining"] = left
    if left["endpoints"] or left["templates"]:
        raise Stop(f"cleanup incomplete: {left}")
    done["nothing_to_do"] = not (done["endpoints_deleted"] or done["templates_deleted"])
    if sf.exists():
        sf.rename(sf.with_suffix(f".deleted-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"))
    return done


def status(cfg: dict, pv: Provider) -> dict:
    e = cfg["endpoint"]
    eps = pv.endpoints_named(e["name"])
    if not eps:
        return {"endpoint": None, "note": "no endpoint with the configured name"}
    eid = eps[0]["id"]
    live = pv.endpoint_settings(eid)
    return {"endpoint_id": eid, "problems": check_settings(live, cfg) + _deploy_only_problems(live, cfg, live.get("templateId")),
            "health": pv.health(eid), "workers_listed": len(live.get("workers") or [])}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "preflight", "deploy", "update", "status", "cleanup"])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    ap.add_argument("--authorize-deploy", action="store_true")
    ap.add_argument("--authorize-update", action="store_true")
    ap.add_argument("--yes", action="store_true", help="cleanup: confirm deletion")
    a = ap.parse_args(argv)
    try:
        cfg = load(a.config)
        campaign = load_campaign(a.campaign) if a.campaign else None
        if a.command == "plan":
            out = plan(cfg, campaign)
        else:
            pv = Provider(os.environ.get(KEY_ENV, ""))
            out = {"preflight": lambda: preflight(cfg, pv, campaign),
                   "deploy": lambda: deploy(cfg, pv, campaign, a.authorize_deploy),
                   "update": lambda: update(cfg, pv, a.authorize_update),
                   "status": lambda: status(cfg, pv),
                   "cleanup": lambda: cleanup(cfg, pv, a.yes)}[a.command]()
    except Stop as exc:
        print(f"STOP: {redact(str(exc), os.environ.get(KEY_ENV, ''))}", file=sys.stderr)
        return 1
    print(redact(json.dumps(out, indent=2), os.environ.get(KEY_ENV, "")))
    if a.command == "preflight" and out.get("problems"):
        return 1
    if a.command == "deploy":
        print(f"\nEndpoint ready (no worker started). Add to .env:  RUNPOD_ENDPOINT_ID={out['endpoint_id']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

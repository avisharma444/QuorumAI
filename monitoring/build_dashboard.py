"""Generates monitoring/grafana/dashboards/councilai.json.

Every panel queries a metric the Go backend actually exports (see
go-backend/internal/metrics). Quantiles aggregate buckets with sum by (le),
which the reference dashboard omitted. Run: python3 monitoring/build_dashboard.py
"""
import json
import pathlib

DS = {"type": "prometheus", "uid": "prometheus"}
panels = []


def panel(title, kind, targets, unit="short", w=12, h=8, desc="", **extra):
    pid = len(panels) + 1
    x = 0 if sum(p["gridPos"]["w"] for p in panels) % 24 == 0 else 12
    p = {
        "id": pid, "title": title, "type": kind, "datasource": DS, "description": desc,
        "gridPos": {"h": h, "w": w, "x": x if w == 12 else 0, "y": 0},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}},
        "targets": [{"datasource": DS, "expr": e, "legendFormat": l, "refId": chr(65 + i)} for i, (e, l) in enumerate(targets)],
    }
    p.update(extra)
    panels.append(p)


def q(expr_tmpl, quantiles=(0.5, 0.95, 0.99), by="", legend=""):
    return [(f"histogram_quantile({x}, sum by (le{', ' + by if by else ''}) ({expr_tmpl}))",
             f"p{int(x * 100)}" + (f" {legend}" if legend else "")) for x in quantiles]


# ── Reference panels (corrected queries) ─────────────────────────────
panel("Request rate by route and status (req/s)", "timeseries",
      [("sum by (route, status) (rate(councilai_request_count_total[1m]))", "{{route}} {{status}}")], unit="reqps")
panel("Query latency p50 / p95 / p99", "timeseries",
      q('rate(councilai_request_latency_seconds_bucket{route="/api/v1/query"}[5m])'), unit="s",
      desc="End-to-end /api/v1/query duration, including SSE streams until they end.")
panel("Council response time by strategy (p50 / p95)", "timeseries",
      q("rate(councilai_council_response_seconds_bucket[5m])", (0.5, 0.95), by="strategy", legend="{{strategy}}"), unit="s")
panel("Cache hit ratio (L1 exact, L2 semantic)", "timeseries", [
    ('sum(rate(councilai_cache_operations_total{level="l1_exact",result="hit"}[5m])) / '
     'sum(rate(councilai_cache_operations_total{level="l1_exact",result=~"hit|miss"}[5m]))', "L1 hits / L1 lookups"),
    ('sum(rate(councilai_cache_operations_total{level="l2_semantic",result="hit"}[5m])) / '
     'sum(rate(councilai_cache_operations_total{level="l2_semantic",result=~"hit|miss"}[5m]))', "L2 hits / L2 lookups"),
], unit="percentunit", desc="Ratios over lookups actually performed; bypassed, disabled and ineligible lookups are excluded.")
panel("Chairman synthesis invocations", "stat",
      [("sum by (result) (increase(councilai_chairman_synthesis_count_total[$__range]))", "{{result}}")])
panel("Model request failures by stage", "stat",
      [("sum by (stage) (increase(councilai_llm_failure_count_total[$__range]))", "{{stage}}")])

# ── Extensions ───────────────────────────────────────────────────────
panel("Council stage duration p95", "timeseries",
      q("rate(councilai_council_stage_seconds_bucket[5m])", (0.95,), by="stage", legend="{{stage}}"), unit="s")
panel("Model request duration p95 by stage", "timeseries",
      q("rate(councilai_llm_request_seconds_bucket[5m])", (0.95,), by="stage", legend="{{stage}}"), unit="s")
panel("Answer source (tier) distribution", "piechart",
      [("sum by (tier) (increase(councilai_query_total[$__range]))", "{{tier}}")],
      desc="none = models ran; l1_exact / l2_semantic = served from cache.")
panel("Cache lookup outcomes (per s)", "timeseries",
      [("sum by (level, result) (rate(councilai_cache_operations_total[5m]))", "{{level}} {{result}}")], unit="reqps")
panel("Cache lookup latency p95", "timeseries",
      q("rate(councilai_cache_lookup_seconds_bucket[5m])", (0.5, 0.95), by="level", legend="{{level}}"), unit="s")
panel("Redis circuit breaker: state and transitions", "timeseries", [
    ("councilai_circuit_breaker_state", "state {{breaker}} (0 closed, 1 half-open, 2 open)"),
    ("sum by (to) (increase(councilai_circuit_breaker_transitions_total[5m]))", "transitions to {{to}} (5m)")],
    desc="The state gauge is sampled every 15 s and can miss a short outage; the transition counter cannot.")
panel("HTTP 5xx ratio", "timeseries", [
    ('sum(rate(councilai_request_count_total{status=~"5.."}[5m])) / sum(rate(councilai_request_count_total[5m]))', "5xx / all")],
    unit="percentunit")
panel("4xx / 499 / 5xx responses by status", "timeseries",
      [('sum by (status) (rate(councilai_request_count_total{status=~"4..|5.."}[5m]))', "{{status}}")], unit="reqps",
      desc="401 auth, 404 unknown/unauthorized document, 429 rate limited, 499 client disconnected, 503 dependency unavailable.")
panel("Provider-reported tokens", "timeseries",
      [("sum by (stage, type) (increase(councilai_llm_tokens_total[5m]))", "{{stage}} {{type}}")],
      desc="Only tokens reported by providers. Mock models report none, so this panel is empty in mock mode.")
panel("In-flight HTTP requests", "timeseries", [("councilai_http_requests_in_flight", "in flight")])
panel("Queries by executed strategy and outcome (per s)", "timeseries",
      [("sum by (strategy, outcome) (rate(councilai_query_total[5m]))", "{{strategy}} {{outcome}}")], unit="reqps")
panel("Rate-limit rejections (per s)", "timeseries",
      [("sum by (limiter, reason) (rate(councilai_rate_limit_rejections_total[5m]))", "{{limiter}} {{reason}}")], unit="reqps")
panel("RAG service latency p95 by endpoint", "timeseries",
      q("rate(councilai_rag_request_seconds_bucket[5m])", (0.5, 0.95), by="endpoint", legend="{{endpoint}}"), unit="s")
panel("Dependency errors (per s)", "timeseries",
      [("sum by (dependency, operation) (rate(councilai_dependency_errors_total[5m]))", "{{dependency}} {{operation}}")], unit="reqps")
panel("Router decisions", "timeseries",
      [("sum by (strategy, fallback) (rate(councilai_router_decisions_total[5m]))", "{{strategy}} fallback={{fallback}}")], unit="reqps")
panel("Client disconnects and trace export errors", "timeseries", [
    ("sum by (route) (rate(councilai_http_client_disconnects_total[5m]))", "disconnects {{route}}"),
    ("rate(councilai_telemetry_errors_total[5m])", "trace export errors")], unit="reqps")

# Lay out two panels per row.
for i, p in enumerate(panels):
    p["gridPos"] = {"h": 8, "w": 12, "x": 0 if i % 2 == 0 else 12, "y": (i // 2) * 8}

dashboard = {
    "uid": "councilai-overview", "title": "CouncilAI", "tags": ["councilai"], "timezone": "browser",
    "schemaVersion": 39, "version": 1, "editable": False, "refresh": "15s",
    "time": {"from": "now-30m", "to": "now"}, "panels": panels,
}
out = pathlib.Path(__file__).parent / "grafana" / "dashboards" / "councilai.json"
out.write_text(json.dumps(dashboard, indent=2) + "\n")
print(f"wrote {out} ({len(panels)} panels)")

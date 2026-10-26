"""Summarize k6 results and container resource samples into Markdown.

Usage: python3 tests/load/summarize.py tests/load/results/<label> [...]
Reads every <profile>-<timestamp>.json written by load_test.js plus the
docker_stats.csv / containers_*.txt written by the runner. Prints tables;
nothing is estimated or filled in: missing values print as "n/a".
"""
import csv
import json
import pathlib
import re
import statistics
import sys
from collections import defaultdict


def v(data, metric, stat):
    m = data.get("metrics", {}).get(metric)
    if not m:
        return None
    return m.get("values", {}).get(stat)


def f(x, digits=1, pct=False):
    if x is None:
        return "n/a"
    return f"{x * 100:.{digits}f}%" if pct else f"{x:.{digits}f}"


def mem_mib(s):
    num, unit = re.match(r"([\d.]+)\s*([KMG]i?B)", s).groups()
    return float(num) * {"KiB": 1 / 1024, "MiB": 1, "GiB": 1024, "kB": 1 / 1000, "MB": 1, "GB": 1000}.get(unit, 1)


def main(dirs):
    rows = []
    for d in dirs:
        for p in sorted(pathlib.Path(d).glob("*.json")):
            prof = p.name.split("-")[0]
            data = json.loads(p.read_text())
            rows.append((prof, p.name, data))
    print("| profile | run | queries | req/s | p50 ms | p95 ms | p99 ms | cold p95 | L1-hit p95 | L2-hit p95 | hit rate | app fail | 429 | 503 | thresholds |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    by = defaultdict(list)
    for prof, name, data in rows:
        th = [(n, all(t["ok"] for t in m["thresholds"].values())) for n, m in data["metrics"].items() if m.get("thresholds") and n != "http_req_failed"]
        thr = "all pass" if th and all(ok for _, ok in th) else ", ".join(f"FAIL {n}" for n, ok in th if not ok) or "-"
        dur = data["metrics"]["http_req_duration"]["values"]
        by[prof].append(data)
        print(f"| {prof} | {name[len(prof) + 1:len(prof) + 20]} | {f(v(data, 'iterations', 'count'), 0)} | {f(v(data, 'http_reqs', 'rate'), 2)} | "
              f"{f(dur.get('med'))} | {f(dur.get('p(95)'))} | {f(dur.get('p(99)'))} | {f(v(data, 'latency_cold_ms', 'p(95)'))} | "
              f"{f(v(data, 'latency_l1_hit_ms', 'p(95)'))} | {f(v(data, 'latency_l2_hit_ms', 'p(95)'))} | {f(v(data, 'cache_hit_rate', 'rate'), 1, True)} | "
              f"{f(v(data, 'app_failure_rate', 'rate'), 2, True)} | {f(v(data, 'status_429', 'count'), 0)} | {f(v(data, 'status_503', 'count'), 0)} | {thr} |")
    print("\nRepeat spread (p95 of all requests, ms):")
    for prof, runs in by.items():
        p95 = [r["metrics"]["http_req_duration"]["values"]["p(95)"] for r in runs]
        spread = f"min {min(p95):.1f}, max {max(p95):.1f}" + (f", stdev {statistics.stdev(p95):.1f}" if len(p95) > 1 else "")
        print(f"- {prof}: {len(p95)} run(s), {spread}")
    for d in dirs:
        stats = pathlib.Path(d) / "docker_stats.csv"
        if not stats.exists():
            continue
        cpu, mem = defaultdict(list), defaultdict(list)
        for row in csv.reader(stats.open()):
            if len(row) < 4:
                continue
            _, name, c, m = row[:4]
            svc = name.replace("councilai-reproduced-", "").rsplit("-", 1)[0]
            cpu[svc].append(float(c.rstrip("%")))
            mem[svc].append(mem_mib(m.split("/")[0].strip()))
        print(f"\nContainer resources during {d} ({len(next(iter(cpu.values()), []))} samples, 5 s apart):")
        print("| service | CPU avg % | CPU max % | mem avg MiB | mem max MiB |")
        print("|---|---|---|---|---|")
        for svc in sorted(cpu):
            print(f"| {svc} | {statistics.mean(cpu[svc]):.1f} | {max(cpu[svc]):.1f} | {statistics.mean(mem[svc]):.0f} | {max(mem[svc]):.0f} |")
        for fn in ("containers_before.txt", "containers_after.txt"):
            p = pathlib.Path(d) / fn
            if p.exists():
                print(f"\n{fn}:\n" + p.read_text().strip())


if __name__ == "__main__":
    main(sys.argv[1:])

"""Check the OpenRouter key's provider-side spending limit before a paid run.

Calls GET /api/v1/key (key metadata; no model is called and nothing is
billed). Prints only the limit, usage and remaining credit, never the key or
its label. Exits non-zero unless the key has a limit no higher than
--max-limit-usd and enough remaining credit for --need-usd.

  python -m councileval.openrouter_key --max-limit-usd 2.00 --need-usd 2.00
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

URL = "https://openrouter.ai/api/v1/key"


def evaluate(data: dict, max_limit: float, need: float) -> list[str]:
    problems = []
    limit = data.get("limit")
    if limit is None:
        problems.append("key has no spending limit; set one of at most $%.2f in the OpenRouter dashboard" % max_limit)
    elif not (0 < float(limit) <= max_limit):
        problems.append(f"key limit ${float(limit):.2f} is above the approved ${max_limit:.2f}")
    remaining = data.get("limit_remaining")
    if remaining is not None and float(remaining) < need:
        problems.append(f"only ${float(remaining):.4f} of the key limit remains; need ${need:.2f}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-limit-usd", type=float, required=True)
    ap.add_argument("--need-usd", type=float, required=True)
    a = ap.parse_args(argv)
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        print("OPENROUTER_API_KEY is not set")
        return 2
    req = urllib.request.Request(URL, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read()).get("data") or {}
    except urllib.error.HTTPError as e:
        print(f"key check failed: HTTP {e.code}")
        return 2
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        print(f"key check failed: {type(e).__name__}")
        return 2
    shown = {k: data.get(k) for k in ("limit", "limit_remaining", "limit_reset", "usage", "usage_daily", "is_free_tier", "include_byok_in_limit")}
    print(json.dumps(shown, indent=2))
    problems = evaluate(data, a.max_limit_usd, a.need_usd)
    for p in problems:
        print("FAIL:", p)
    if not problems:
        print(f"OK: provider-side limit ${float(data['limit']):.2f} <= approved ${a.max_limit_usd:.2f}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

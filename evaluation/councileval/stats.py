"""Uncertainty estimates and paired tests. All resampling is seeded."""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

DEFAULT_SEED = 20261027
DEFAULT_RESAMPLES = 10_000


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[Optional[float], Optional[float]]:
    """95% Wilson score interval for a proportion (good for small n and
    proportions near 0 or 1, where the normal approximation fails)."""
    if n == 0:
        return None, None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_mean_ci(values: Sequence[float], seed: int = DEFAULT_SEED, resamples: int = DEFAULT_RESAMPLES,
                      level: float = 0.95) -> tuple[Optional[float], Optional[float]]:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if v.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(resamples, v.size))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], seed: int = DEFAULT_SEED,
                          resamples: int = DEFAULT_RESAMPLES, level: float = 0.95) -> dict:
    """Mean of (b - a) over the same items, with a percentile bootstrap CI
    that resamples items (keeping each pair together)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return {"n": 0, "mean_a": None, "mean_b": None, "diff": None, "ci": (None, None), "relative": None}
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)
    d = y - x
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(resamples, d.size))
    boot = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, (1 - level) / 2), np.quantile(boot, 1 - (1 - level) / 2)
    mean_a = float(x.mean())
    return {"n": int(d.size), "mean_a": mean_a, "mean_b": float(y.mean()), "diff": float(d.mean()),
            "ci": (float(lo), float(hi)), "relative": (float(d.mean()) / mean_a) if mean_a else None}


def mcnemar_exact(a_correct: Sequence[Optional[bool]], b_correct: Sequence[Optional[bool]]) -> dict:
    """Exact (binomial) McNemar test on paired binary outcomes. Only the
    discordant pairs carry information: b01 = A wrong & B right, b10 = A
    right & B wrong. Two-sided p-value."""
    pairs = [(x, y) for x, y in zip(a_correct, b_correct) if x is not None and y is not None]
    b01 = sum(1 for x, y in pairs if not x and y)
    b10 = sum(1 for x, y in pairs if x and not y)
    n = b01 + b10
    if n == 0:
        p = 1.0
    else:
        k = min(b01, b10)
        tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
        p = min(1.0, 2 * tail)
    return {"n_pairs": len(pairs), "a_only": b10, "b_only": b01, "p_value": p}


def describe_difference(result: dict, alpha: float = 0.05) -> str:
    """Plain-language verdict that never over-claims."""
    if result.get("n", 0) == 0 or result.get("diff") is None:
        return "not measured"
    lo, hi = result["ci"]
    if lo > 0:
        return f"improvement (95% CI {lo:+.3f} to {hi:+.3f})"
    if hi < 0:
        return f"regression (95% CI {lo:+.3f} to {hi:+.3f})"
    return f"no reliable difference (95% CI {lo:+.3f} to {hi:+.3f} includes 0)"

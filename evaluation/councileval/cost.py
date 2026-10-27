"""Usage accounting, pricing and budget enforcement.

Three kinds of numbers are kept apart and never mixed:

* reported   token counts the provider returned (llm.Usage in responses);
* estimated  cost = reported tokens x a VERIFIED price from the pricing
             table (source URL + effective date); never a guessed price;
* unmeasured calls without reported usage, or models without a verified
             price. These are counted, not filled in.

Mock models report no usage: actual spend is 0 and any "what would this
cost with real models" figure stays unmeasured.

The BudgetGuard enforces a request limit and a spend limit before every
request using a conservative worst-case estimate, and stops the run as soon
as the next request could exceed either limit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

# Model calls one request makes, by executed strategy (router call added for "auto").
CALLS_PER_STRATEGY = {"direct": 1, "council": 7, "council_deep": 9}  # deep: 7 + reflection + at most 1 revision


class BudgetExceeded(RuntimeError):
    pass


class PricingError(RuntimeError):
    pass


@dataclass(frozen=True)
class Price:
    model: str
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    source: str
    effective_date: str
    verified: bool


def load_pricing(path: Path) -> dict[str, Price]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    out = {}
    for e in data.get("models", []) or []:
        p = Price(model=e["model"], input_usd_per_mtok=float(e["input_usd_per_mtok"]),
                  output_usd_per_mtok=float(e["output_usd_per_mtok"]), source=e.get("source", ""),
                  effective_date=str(e.get("effective_date", "")), verified=bool(e.get("verified", False)))
        if p.verified and (not p.source.startswith("http") or not p.effective_date):
            raise PricingError(f"{p.model}: a verified price needs a source URL and an effective date")
        if p.input_usd_per_mtok < 0 or p.output_usd_per_mtok < 0:
            raise PricingError(f"{p.model}: negative price")
        out[p.model] = p
    return out


def require_verified(pricing: dict[str, Price], models: list[str]) -> None:
    """Real-model runs need a verified price for every configured model."""
    missing = [m for m in models if m not in pricing or not pricing[m].verified]
    if missing:
        raise PricingError("no verified price for: " + ", ".join(sorted(set(missing))))


@dataclass
class UsageLedger:
    calls: int = 0
    calls_with_usage: int = 0
    calls_without_usage: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0
    cost_unmeasured_calls: int = 0
    failed_calls: int = 0
    by_model: dict = field(default_factory=dict)

    def add_call(self, model: str, usage: Optional[dict], pricing: dict[str, Price], failed: bool = False) -> None:
        self.calls += 1
        if failed:
            self.failed_calls += 1
        m = self.by_model.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "without_usage": 0})
        m["calls"] += 1
        if not usage or not any(usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")):
            self.calls_without_usage += 1
            m["without_usage"] += 1
            self.cost_unmeasured_calls += 1
            return
        self.calls_with_usage += 1
        pt, ct = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        m["prompt_tokens"] += pt
        m["completion_tokens"] += ct
        price = pricing.get(model)
        if price and price.verified:
            self.estimated_cost_usd += pt / 1e6 * price.input_usd_per_mtok + ct / 1e6 * price.output_usd_per_mtok
        else:
            self.cost_unmeasured_calls += 1

    def summary(self, mock: bool) -> dict:
        return {
            "actual_api_spend_usd": 0.0 if mock else None,  # provider billing is not readable from the API
            "actual_api_spend_note": "mock models: no provider was called" if mock else
            "not available from provider responses; see estimated_cost_usd and the provider's billing console",
            "reported_prompt_tokens": self.prompt_tokens, "reported_completion_tokens": self.completion_tokens,
            "calls": self.calls, "calls_with_reported_usage": self.calls_with_usage,
            "calls_without_reported_usage": self.calls_without_usage, "failed_calls": self.failed_calls,
            "estimated_cost_usd": None if (mock or self.calls_with_usage == 0) else round(self.estimated_cost_usd, 6),
            "cost_unmeasured_calls": self.cost_unmeasured_calls,
            "hypothetical_real_model_cost": "unmeasured" if mock else None,
            "by_model": self.by_model,
        }


def estimate_tokens(text_chars: int) -> int:
    """Conservative character-to-token estimate (~3 characters per token for
    English; most tokenizers average ~4, so this over-estimates)."""
    return math.ceil(text_chars / 3)


@dataclass
class RequestEstimate:
    calls: int
    input_tokens: int
    output_tokens: int

    def cost(self, price_in: float, price_out: float) -> float:
        return self.input_tokens / 1e6 * price_in + self.output_tokens / 1e6 * price_out


def worst_case_request(strategy: str, question_chars: int, evidence_chars: int, max_output_tokens: int,
                       routed: bool) -> RequestEstimate:
    """Upper-bound estimate for one /query request. Reviews and the chairman
    read every candidate, so their inputs include all candidate outputs."""
    calls = CALLS_PER_STRATEGY["council_deep" if strategy == "auto" else strategy]
    base = estimate_tokens(question_chars + evidence_chars) + 600  # + system prompt and formatting
    if strategy == "direct":
        inp = base
    else:
        cands = 3 * max_output_tokens
        inp = 3 * base + 3 * (base + cands) + (base + cands + 3 * max_output_tokens)
        if strategy in ("council_deep", "auto"):
            inp += 2 * (base + max_output_tokens)
    out = calls * max_output_tokens
    if routed or strategy == "auto":
        calls += 1
        inp += estimate_tokens(question_chars) + 1200
        out += 200
    return RequestEstimate(calls, inp, out)


class BudgetGuard:
    """Refuses a request when it could exceed the request or spend limit.

    Spend is tracked conservatively: a finished request adds its estimated
    cost from reported usage, but if any call lacked usage the request's
    worst-case estimate is charged instead (fail closed)."""

    def __init__(self, max_requests: int, max_spend_usd: float, price_in: float, price_out: float):
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        if max_spend_usd < 0:
            raise ValueError("max_spend_usd must be >= 0")
        self.max_requests, self.max_spend = max_requests, max_spend_usd
        self.price_in, self.price_out = price_in, price_out
        self.requests = 0
        self.spent = 0.0

    def check(self, estimate: RequestEstimate) -> None:
        if self.requests >= self.max_requests:
            raise BudgetExceeded(f"request limit reached ({self.max_requests})")
        worst = estimate.cost(self.price_in, self.price_out)
        if self.spent + worst > self.max_spend:
            raise BudgetExceeded(f"next request could cost up to ${worst:.4f}; spent ${self.spent:.4f} of ${self.max_spend:.2f}")

    def record(self, estimate: RequestEstimate, reported_cost: Optional[float]) -> None:
        self.requests += 1
        self.spent += reported_cost if reported_cost is not None else estimate.cost(self.price_in, self.price_out)

    def state(self) -> dict:
        return {"requests": self.requests, "max_requests": self.max_requests, "charged_usd": round(self.spent, 6),
                "max_spend_usd": self.max_spend}


def today() -> str:
    return date.today().isoformat()

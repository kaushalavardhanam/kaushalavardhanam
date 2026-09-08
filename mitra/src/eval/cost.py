"""Order-of-magnitude Bedrock token cost estimates (issue #7).

Rates are public list prices recorded 2026-09-08 and will drift.
They are for evaluation tables only — not billing.
"""

from __future__ import annotations

# USD per 1M tokens. Sources: https://aws.amazon.com/bedrock/pricing/
# Confirm at run time before using for budget decisions.
USD_PER_MTOK = {
    "amazon.nova-lite": {"input": 0.06, "output": 0.24},
    "amazon.nova-pro": {"input": 0.80, "output": 3.20},
    "amazon.nova-premier": {"input": 2.50, "output": 12.50},
    "anthropic.claude-sonnet-4": {"input": 3.00, "output": 15.00},
    "anthropic.claude-haiku-4": {"input": 1.00, "output": 5.00},
    "qwen": {"input": 0.0, "output": 0.0},
}


def rate_for(model_id: str) -> dict[str, float] | None:
    mid = (model_id or "").lower()
    for key, rate in USD_PER_MTOK.items():
        if key in mid:
            return rate
    return None


def estimate_usd(model_id: str, input_tokens: int | None,
                 output_tokens: int | None) -> float | None:
    rate = rate_for(model_id)
    if rate is None or input_tokens is None or output_tokens is None:
        return None
    return round(
        (input_tokens / 1_000_000) * rate["input"]
        + (output_tokens / 1_000_000) * rate["output"],
        6,
    )

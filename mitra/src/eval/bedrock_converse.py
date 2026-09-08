"""Direct Bedrock Converse helper for controlled (Mode B) evaluation.

Uses the standard AWS credential chain. Does not import or contact Ollama.
Strands is not required — evaluation scripts can score models even when the
agent extra is missing.
"""

from __future__ import annotations

import time
from typing import Any

from mitra.agent.errors import ProviderError
from mitra.agent.provider import map_provider_exception, resolve_region


def converse_text(
    *,
    model_id: str,
    user: str,
    system: str,
    region: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 256,
    timeout_s: float = 45,
    images: list[bytes] | None = None,
) -> dict[str, Any]:
    try:
        import boto3
        from botocore.config import Config as BotocoreConfig
    except ImportError as e:
        raise ProviderError(
            "boto3 is required to call Bedrock.",
            code="missing_dependency",
            provider="bedrock",
            model_id=model_id,
            actionable="pip install 'mitra[bedrock]'",
        ) from e

    region = region or resolve_region({"region": region})
    client = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=BotocoreConfig(read_timeout=timeout_s, connect_timeout=5,
                              retries={"max_attempts": 2, "mode": "standard"}),
    )
    content: list[dict] = []
    for jpeg in images or []:
        content.append({"image": {"format": "jpeg", "source": {"bytes": jpeg}}})
    content.append({"text": user})
    t0 = time.monotonic()
    try:
        resp = client.converse(
            modelId=model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
        )
    except Exception as e:
        raise map_provider_exception(e, provider="bedrock", model_id=model_id,
                                     region=region) from e
    elapsed = time.monotonic() - t0
    parts = resp.get("output", {}).get("message", {}).get("content") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    usage = resp.get("usage") or {}
    return {
        "text": text.strip(),
        "model_id": model_id,
        "region": region,
        "latency_s": round(elapsed, 3),
        "input_tokens": usage.get("inputTokens"),
        "output_tokens": usage.get("outputTokens"),
        "stop_reason": resp.get("stopReason"),
    }

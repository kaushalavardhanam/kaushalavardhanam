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
    history: list[dict[str, Any]] | None = None,
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
    messages = list(history or [])
    messages.append({"role": "user", "content": content})
    t0 = time.monotonic()
    inference: dict[str, Any] = {"maxTokens": max_tokens}
    if temperature is not None:
        inference["temperature"] = temperature
    try:
        resp = client.converse(
            modelId=model_id,
            system=[{"text": system}],
            messages=messages,
            inferenceConfig=inference,
        )
    except Exception as e:
        text = str(e)
        if (
            temperature is not None
            and "doesn't support the temperature field" in text
        ):
            inference.pop("temperature", None)
            try:
                resp = client.converse(
                    modelId=model_id,
                    system=[{"text": system}],
                    messages=messages,
                    inferenceConfig=inference,
                )
            except Exception as retry_exc:
                raise map_provider_exception(
                    retry_exc, provider="bedrock", model_id=model_id, region=region
                ) from retry_exc
        else:
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
        "assistant_message": {"role": "assistant", "content": parts or [{"text": text}]},
    }

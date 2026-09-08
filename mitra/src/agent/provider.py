"""LLM/VLM provider construction (DESIGN §1.5, issue #7).

Bedrock mode uses the standard AWS credential-provider chain and never
imports, starts, or contacts Ollama. Local Ollama/Qwen is constructed only
when ``provider: ollama``. Fallback to another provider is opt-in and
explicit — a Bedrock failure never silently changes models.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .errors import ProviderError

logger = logging.getLogger("mitra")

SUPPORTED_PROVIDERS = ("ollama", "anthropic", "bedrock")


def provider_name(cfg: dict) -> str:
    return str(cfg.get("provider") or "ollama").strip().lower()


def model_id(cfg: dict) -> str:
    return str(cfg.get("id") or cfg.get("model_id") or "")


def resolve_region(cfg: dict) -> str | None:
    """Region from config, else AWS_REGION / AWS_DEFAULT_REGION. Never hard-coded."""
    explicit = cfg.get("region") or cfg.get("region_name")
    if explicit:
        return str(explicit)
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")


def describe_llm(cfg: dict) -> dict[str, Any]:
    """Safe identity for logs — no credentials, prompts, or images."""
    name = provider_name(cfg)
    info = {
        "provider": name,
        "model_id": model_id(cfg),
        "temperature": cfg.get("temperature", 0.3),
    }
    if name == "bedrock":
        info["region"] = resolve_region(cfg)
        info["timeout_s"] = cfg.get("timeout_s", 45)
        info["max_retries"] = cfg.get("max_retries", 2)
        info["max_tokens"] = cfg.get("max_tokens", 256)
        info["streaming"] = bool(cfg.get("streaming", False))
        info["ollama_contacted"] = False
    elif name == "ollama":
        info["host"] = cfg.get("host", "http://localhost:11434")
        info["keep_alive"] = cfg.get("keep_alive", "30m")
    return info


def assert_no_ollama_in_bedrock(cfg: dict) -> None:
    if provider_name(cfg) == "bedrock" and cfg.get("host"):
        # host is ignored, not contacted — keep a breadcrumb for operators
        logger.info("bedrock mode: ignoring ollama host %s (will not be contacted)",
                    cfg.get("host"))


def fallback_config(cfg: dict) -> dict | None:
    """Return an explicit fallback LLM config, or None if fallback is off."""
    fb = cfg.get("fallback") or {}
    if not fb or not fb.get("enabled"):
        return None
    if not fb.get("provider") or not (fb.get("id") or fb.get("model_id")):
        raise ProviderError(
            "LLM fallback is enabled but provider/id is missing.",
            code="fallback_misconfigured",
            provider=provider_name(cfg),
            model_id=model_id(cfg),
            actionable="Set models.llm.fallback.provider and models.llm.fallback.id, "
                       "or set fallback.enabled: false.",
        )
    return {
        "provider": fb["provider"],
        "id": fb.get("id") or fb.get("model_id"),
        "temperature": fb.get("temperature", cfg.get("temperature", 0.3)),
        "host": fb.get("host", "http://localhost:11434"),
        "region": fb.get("region"),
        "timeout_s": fb.get("timeout_s", cfg.get("timeout_s", 45)),
        "max_retries": fb.get("max_retries", 1),
        "max_tokens": fb.get("max_tokens", cfg.get("max_tokens", 256)),
    }


def make_model(cfg: dict):
    """Construct the Strands model for ``cfg``. Imports only the selected provider."""
    name = provider_name(cfg)
    if name not in SUPPORTED_PROVIDERS:
        raise ProviderError(
            f"Unknown LLM provider {name!r}.",
            code="unknown_provider",
            provider=name,
            model_id=model_id(cfg),
            actionable=f"Set models.llm.provider to one of {SUPPORTED_PROVIDERS}.",
        )
    if name == "ollama":
        return _make_ollama(cfg)
    if name == "anthropic":
        return _make_anthropic(cfg)
    return _make_bedrock(cfg)


def _make_ollama(cfg: dict):
    try:
        from strands.models.ollama import OllamaModel
    except ImportError as e:
        raise ProviderError(
            "strands-agents is required for the Ollama provider.",
            code="missing_dependency",
            provider="ollama",
            model_id=model_id(cfg),
            actionable="Install with: pip install 'mitra[agent]'",
        ) from e
    mid = model_id(cfg) or "qwen3-vl:8b-instruct"
    logger.info("LLM provider=ollama model=%s host=%s", mid,
                cfg.get("host", "http://localhost:11434"))
    return OllamaModel(
        host=cfg.get("host", "http://localhost:11434"),
        model_id=mid,
        temperature=cfg.get("temperature", 0.3),
        keep_alive=cfg.get("keep_alive", "30m"),
        additional_args={"think": cfg.get("think", False)},
    )


def _make_anthropic(cfg: dict):
    mid = model_id(cfg)
    if not mid:
        raise ProviderError(
            "Anthropic provider requires models.llm.id.",
            code="missing_model_id",
            provider="anthropic",
            actionable="Set models.llm.id to an Anthropic model id.",
        )
    try:
        from strands.models.anthropic import AnthropicModel
    except ImportError as e:
        raise ProviderError(
            "strands-agents is required for the Anthropic provider.",
            code="missing_dependency",
            provider="anthropic",
            model_id=mid,
            actionable="Install with: pip install 'mitra[agent]'",
        ) from e
    logger.info("LLM provider=anthropic model=%s (Ollama will not be contacted)", mid)
    return AnthropicModel(model_id=mid)


def _make_bedrock(cfg: dict):
    mid = model_id(cfg)
    if not mid:
        raise ProviderError(
            "Bedrock provider requires models.llm.id.",
            code="missing_model_id",
            provider="bedrock",
            actionable="Set models.llm.id to a Bedrock model or inference-profile id. "
                       "Do not commit credentials or a hard-coded account-specific ARN.",
        )
    region = resolve_region(cfg)
    assert_no_ollama_in_bedrock(cfg)
    try:
        from botocore.config import Config as BotocoreConfig
        from strands.models import BedrockModel
    except ImportError as e:
        raise ProviderError(
            "Bedrock mode needs strands-agents and boto3/botocore.",
            code="missing_dependency",
            provider="bedrock",
            model_id=mid,
            region=region,
            actionable="Install with: pip install 'mitra[bedrock]'",
        ) from e

    timeout_s = float(cfg.get("timeout_s", 45))
    connect_timeout_s = float(cfg.get("connect_timeout_s", 5))
    max_retries = int(cfg.get("max_retries", 2))
    boto_cfg = BotocoreConfig(
        connect_timeout=connect_timeout_s,
        read_timeout=timeout_s,
        retries={"max_attempts": max_retries, "mode": "standard"},
    )
    kwargs: dict[str, Any] = {
        "model_id": mid,
        "temperature": cfg.get("temperature", 0.3),
        "max_tokens": cfg.get("max_tokens", 256),
        "streaming": bool(cfg.get("streaming", False)),
        "boto_client_config": boto_cfg,
    }
    if region:
        kwargs["region_name"] = region
    if cfg.get("top_p") is not None:
        kwargs["top_p"] = cfg["top_p"]

    logger.info(
        "LLM provider=bedrock model=%s region=%s timeout_s=%s retries=%s "
        "(Ollama/Qwen will not be started, loaded, or contacted)",
        mid, region or "(default credential-chain region)", timeout_s, max_retries,
    )
    try:
        return BedrockModel(**kwargs)
    except Exception as e:
        raise map_provider_exception(e, provider="bedrock", model_id=mid,
                                     region=region) from e


def bedrock_credential_status() -> dict[str, Any]:
    """Presence check only — never returns secret material."""
    try:
        import boto3
    except ImportError:
        return {"ok": False, "reason": "boto3 is not installed (pip install 'mitra[bedrock]')"}
    session = boto3.Session()
    creds = session.get_credentials()
    if creds is None:
        return {
            "ok": False,
            "reason": "no AWS credentials in the standard provider chain",
            "hint": "Configure env vars, ~/.aws/credentials, an instance role, or SSO. "
                    "Mitra never accepts hard-coded keys in config.yaml.",
        }
    frozen = creds.get_frozen_credentials()
    return {
        "ok": True,
        "method": getattr(creds, "method", "chain"),
        "region": session.region_name or resolve_region({}),
        "access_key_fp": (frozen.access_key[:4] + "…" + frozen.access_key[-2:]
                          if frozen.access_key and len(frozen.access_key) > 6 else "present"),
    }


def map_provider_exception(exc: BaseException, *, provider: str,
                           model_id: str | None, region: str | None) -> ProviderError:
    """Turn SDK/AWS exceptions into operator-facing ProviderError values."""
    name = type(exc).__name__
    text = str(exc)
    code = getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else None
    code = code or name

    if name in {"NoCredentialsError", "PartialCredentialsError"} or "Unable to locate credentials" in text:
        return ProviderError(
            "AWS credentials were not found.",
            code="no_credentials",
            provider=provider, model_id=model_id, region=region,
            actionable="Use the standard AWS credential-provider chain "
                       "(environment, shared config, or instance/task role). "
                       "Do not put keys in config.yaml.",
        )
    if code in {"UnrecognizedClientException", "InvalidSignatureException"}:
        return ProviderError(
            "AWS rejected the credentials for Bedrock.",
            code="bad_credentials",
            provider=provider, model_id=model_id, region=region,
            actionable="Refresh credentials and confirm the identity can call bedrock:InvokeModel.",
        )
    if code in {"AccessDeniedException", "UnauthorizedOperation"} or "not authorized" in text.lower():
        return ProviderError(
            f"This identity cannot invoke {model_id} in {region or 'the configured region'}.",
            code="access_denied",
            provider=provider, model_id=model_id, region=region,
            actionable="Enable model access in the Bedrock console and grant "
                       "bedrock:InvokeModel (and Converse) on that model/inference profile.",
        )
    if "doesn't support the temperature field" in text:
        return ProviderError(
            f"{model_id} rejects inferenceConfig.temperature.",
            code="unsupported_parameter",
            provider=provider, model_id=model_id, region=region,
            actionable="Omit temperature for this model (GPT-5.6 Sol and some reasoning IDs).",
        )
    if code in {"ResourceNotFoundException"} or "model identifier is invalid" in text.lower():
        return ProviderError(
            f"Model {model_id!r} is not available in {region or 'the configured region'}.",
            code="unsupported_model_or_region",
            provider=provider, model_id=model_id, region=region,
            actionable="Check the exact model or inference-profile id and Region. "
                       "Mitra will not silently switch to another model.",
        )
    if code in {"ThrottlingException", "TooManyRequestsException", "ServiceQuotaExceededException"}:
        return ProviderError(
            f"Bedrock throttled or quota-limited {model_id}.",
            code="throttled",
            provider=provider, model_id=model_id, region=region,
            actionable="Retry later, lower concurrency, or request a quota increase. "
                       "No automatic fallback unless models.llm.fallback.enabled is true.",
        )
    if name in {"ReadTimeoutError", "ConnectTimeoutError", "EndpointConnectionError",
                "ConnectionClosedError"} or "timed out" in text.lower():
        return ProviderError(
            f"Bedrock request to {model_id} timed out or could not connect.",
            code="timeout",
            provider=provider, model_id=model_id, region=region,
            actionable="Check network egress to bedrock-runtime and raise models.llm.timeout_s if needed.",
        )
    if "malformed" in text.lower() or "json" in text.lower() and "parse" in text.lower():
        return ProviderError(
            "The model returned malformed output.",
            code="malformed_output",
            provider=provider, model_id=model_id, region=region,
            actionable="Retry the turn; if it persists, this model is a poor fit for tool/JSON turns.",
        )
    return ProviderError(
        f"Provider {provider} failed ({code}): {text[:240]}",
        code=str(code),
        provider=provider, model_id=model_id, region=region,
        actionable="See the log for the exception; the provider/model was not changed.",
    )

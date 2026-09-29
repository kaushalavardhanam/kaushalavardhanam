"""Bedrock runtime client for the AgentCore decomposer agent.

Wraps ``bedrock-runtime.invoke_model`` and dispatches on the configured model's
provider so the same public ``invoke(prompt, system=..., temperature=...)`` call
works across two request/response schemas:

* **Anthropic** (``anthropic.*`` / ``global.anthropic.*``) — the Anthropic
  Messages API shape (``anthropic_version = bedrock-2023-05-31``) that the
  repo's ``coding-agent/graph_workflow.py`` already uses.
* **OpenAI** (``openai.*`` / ``global.openai.*``) — the OpenAI
  chat-completions shape that Bedrock's OpenAI models accept
  (``messages`` array with ``system``/``user`` roles, ``max_completion_tokens``),
  with the response text at ``choices[0].message.content``.

The model id is a *global CRIS* inference-profile id (the ``global.`` form),
which Bedrock routes to the nearest region; the AgentCore Runtime execution
role (see ``.github/agentcore/terraform/main.tf``) is granted invoke on the FM
and inference-profile ARNs each of the three allowed models resolves to.

Nothing here is secret: the region and model id come from environment variables
that the Terraform sets on the runtime (``BEDROCK_MODEL_ID``), defaulting to
Claude Sonnet 5.5.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class TruncatedResponseError(RuntimeError):
    """Raised when the model stopped because it hit the output-token cap.

    A truncated response is an incomplete document (e.g. a JSON object cut off
    mid-string), so surfacing it as a clear, actionable error is far better than
    letting a downstream ``json.loads`` fail with a cryptic "Unterminated
    string" / "No JSON object found". This is exactly how issue #13 failed when
    max_tokens was 8192.
    """

# The default model is Claude Sonnet 5.5 via its global CRIS inference profile.
# GPT models are opt-in by overriding BEDROCK_MODEL_ID with one of the other
# allowed ids below.
DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-5-5"
DEFAULT_REGION = "us-east-1"
ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Max output tokens per model call. The implement step returns whole file
# contents as JSON string values, so a small cap truncates the JSON mid-string
# and the downstream parser then fails on an incomplete object (this is what
# broke issue #13 at 8192). Claude Sonnet 5.5 supports a far larger output
# window, so default high; callers may still override per-call.
DEFAULT_MAX_TOKENS = 32768

# botocore read timeout for a single invoke_model call. The default is 60s,
# but a single large-output generation (up to DEFAULT_MAX_TOKENS) can
# legitimately take several minutes; once max_tokens was raised the 60s cap
# started raising ReadTimeoutError mid-generation. Give a generous ceiling;
# adaptive retries still handle genuine transient stalls.
BEDROCK_READ_TIMEOUT_SECONDS = 900

# The only model ids the runtime is permitted to invoke. These are the global
# CRIS inference-profile ids the Terraform grants InvokeModel on; invoking via
# the bare foundation-model id is intentionally NOT allowed. Keep this in sync
# with the Bedrock grant in ../terraform/main.tf (local.bedrock_models).
ALLOWED_MODEL_IDS = (
    "global.anthropic.claude-sonnet-5-5",  # DEFAULT — Anthropic Messages schema
    "global.openai.gpt-5.6-sol",           # opt-in — OpenAI chat-completions
    "global.openai.gpt-6-astra",           # opt-in — OpenAI chat-completions
)


def get_model_id() -> str:
    """Resolve the Bedrock model / inference-profile id from the environment.

    The Terraform sets ``BEDROCK_MODEL_ID`` as a runtime environment variable.
    """
    return os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)


def get_region() -> str:
    """Resolve the AWS region.

    Prefers ``AWS_REGION`` (set by the runtime), then ``AWS_DEFAULT_REGION``,
    falling back to us-east-1.
    """
    return (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or DEFAULT_REGION
    )


def _provider_of(model_id: str) -> str:
    """Classify a model id as ``anthropic`` or ``openai`` by its provider token.

    The id may carry a CRIS routing prefix (``global.``/``us.``/``eu.``/``apac.``);
    the provider is the token immediately after any such prefix.
    """
    mid = model_id.lower()
    for prefix in ("global.", "us.", "eu.", "apac."):
        if mid.startswith(prefix):
            mid = mid[len(prefix):]
            break
    if mid.startswith("anthropic."):
        return "anthropic"
    if mid.startswith("openai."):
        return "openai"
    raise ValueError(
        f"Cannot determine provider for model id {model_id!r}; "
        f"expected an anthropic.* or openai.* id."
    )


class BedrockClient:
    """Provider-aware wrapper around ``bedrock-runtime.invoke_model``."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or get_model_id()
        if self.model_id not in ALLOWED_MODEL_IDS:
            raise ValueError(
                f"Unsupported BEDROCK_MODEL_ID {self.model_id!r}. "
                f"Allowed model ids are: {', '.join(ALLOWED_MODEL_IDS)}."
            )
        self.provider = _provider_of(self.model_id)
        self.region_name = region_name or get_region()
        # Retries help with transient Bedrock throttling during a multi
        # sub-task run. read_timeout is raised well above botocore's 60s default
        # because a single large-output generation (up to DEFAULT_MAX_TOKENS)
        # can legitimately take several minutes; the default would raise
        # ReadTimeoutError mid-generation once max_tokens was increased.
        self._client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region_name,
            config=Config(
                retries={"max_attempts": 5, "mode": "adaptive"},
                connect_timeout=15,
                read_timeout=BEDROCK_READ_TIMEOUT_SECONDS,
            ),
        )
        logger.info(
            "BedrockClient ready (model=%s provider=%s region=%s)",
            self.model_id,
            self.provider,
            self.region_name,
        )

    def invoke(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.2,
    ) -> str:
        """Invoke the configured model with a single user prompt.

        The request/response schema is selected from the model's provider; the
        signature is identical across providers so callers (``decomposer.py``)
        need no changes.

        Args:
            prompt: The user message content.
            system: Optional system prompt.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature. NOTE: Bedrock's OpenAI models
                currently accept only the default temperature (1); a non-default
                value is dropped for the OpenAI path rather than sent (which
                would raise a ValidationException).

        Returns:
            The text of the model's response.

        Raises:
            ClientError: On an AWS/Bedrock API error (surfaced to the caller so
                the agent can fail the run rather than silently continue).
            ValueError: If the response contained no text content.
        """
        if self.provider == "anthropic":
            body = self._anthropic_body(
                prompt, system=system, max_tokens=max_tokens, temperature=temperature
            )
        else:  # openai
            body = self._openai_body(
                prompt, system=system, max_tokens=max_tokens, temperature=temperature
            )

        try:
            response = self._client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )
        except ClientError:
            logger.exception("Bedrock invoke_model failed (model=%s)", self.model_id)
            raise

        payload = json.loads(response["body"].read())
        if self.provider == "anthropic":
            self._raise_if_truncated_anthropic(payload, max_tokens)
            return self._extract_anthropic_text(payload)
        self._raise_if_truncated_openai(payload, max_tokens)
        return self._extract_openai_text(payload)

    # --------------------------------------------------------------------- #
    # Anthropic Messages API
    # --------------------------------------------------------------------- #

    @staticmethod
    def _anthropic_body(
        prompt: str,
        *,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        body: dict = {
            "anthropic_version": ANTHROPIC_VERSION,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        # Bedrock's newer Anthropic models (e.g. Claude Sonnet 5.5) reject a
        # non-default temperature with "`temperature` is deprecated for this
        # model"; they accept the default (1) or its omission. Send temperature
        # only when it is that default, and omit it otherwise.
        if abs(temperature - BedrockClient._DEFAULT_TEMPERATURE) < 1e-9:
            body["temperature"] = BedrockClient._DEFAULT_TEMPERATURE
        if system:
            body["system"] = system
        return body

    @staticmethod
    def _raise_if_truncated_anthropic(payload: dict, max_tokens: int) -> None:
        """Fail loudly if the Claude response was cut off at the token cap.

        The Anthropic Messages API reports ``stop_reason == "max_tokens"`` when
        generation stopped because it reached ``max_tokens`` rather than a
        natural end (``end_turn``/``stop_sequence``). In that case the returned
        content is incomplete, so we raise instead of returning a partial body.
        """
        if payload.get("stop_reason") == "max_tokens":
            raise TruncatedResponseError(
                f"BedrockClient: response truncated at max_tokens ({max_tokens}) "
                f"(stop_reason=max_tokens) — increase max_tokens or reduce sub-task size."
            )

    @staticmethod
    def _extract_anthropic_text(payload: dict) -> str:
        """Pull the text out of a Claude Messages API response payload."""
        blocks = payload.get("content") or []
        parts = [b.get("text", "") for b in blocks if b.get("type", "text") == "text"]
        text = "".join(parts).strip()
        if not text:
            raise ValueError("Bedrock response contained no text content")
        return text

    # --------------------------------------------------------------------- #
    # OpenAI chat-completions API
    # --------------------------------------------------------------------- #

    # Both providers on Bedrock currently accept only the default temperature
    # (1); a non-default value returns a ValidationException. Send temperature
    # ONLY when it equals this default, and otherwise omit it.
    _DEFAULT_TEMPERATURE = 1.0

    @classmethod
    def _openai_body(
        cls,
        prompt: str,
        *,
        system: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> dict:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        body: dict = {
            "messages": messages,
            # OpenAI-on-Bedrock uses max_completion_tokens, not max_tokens.
            "max_completion_tokens": max_tokens,
        }
        # Only forward temperature when it is the sole value these models accept.
        if abs(temperature - cls._DEFAULT_TEMPERATURE) < 1e-9:
            body["temperature"] = cls._DEFAULT_TEMPERATURE
        return body

    @staticmethod
    def _raise_if_truncated_openai(payload: dict, max_tokens: int) -> None:
        """Fail loudly if the OpenAI response was cut off at the token cap.

        OpenAI chat-completions reports ``choices[i].finish_reason == "length"``
        when generation stopped at ``max_completion_tokens`` rather than a
        natural ``stop``. The content is then incomplete, so we raise instead of
        returning a partial body.
        """
        for choice in payload.get("choices") or []:
            if choice.get("finish_reason") == "length":
                raise TruncatedResponseError(
                    f"BedrockClient: response truncated at max_tokens ({max_tokens}) "
                    f"(finish_reason=length) — increase max_tokens or reduce sub-task size."
                )

    @staticmethod
    def _extract_openai_text(payload: dict) -> str:
        """Pull the text out of an OpenAI chat-completions response payload."""
        choices = payload.get("choices") or []
        parts = []
        for choice in choices:
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                # Some OpenAI responses use a content-parts array.
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        parts.append(part["text"])
        text = "".join(parts).strip()
        if not text:
            raise ValueError("Bedrock response contained no text content")
        return text

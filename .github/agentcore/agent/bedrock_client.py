"""Bedrock runtime client for the AgentCore decomposer agent.

Wraps ``bedrock-runtime.invoke_model`` for the pinned Anthropic Claude model,
using the Messages API shape (``anthropic_version = bedrock-2023-05-31``) that
the repo's existing ``coding-agent/graph_workflow.py`` already uses. The model
id is a *global CRIS* inference-profile id (``global.anthropic.claude-opus-5``);
Bedrock routes it to the nearest region, and the AgentCore Runtime execution
role (see the Terraform in ``.github/agentcore/terraform/main.tf``) is granted
invoke on the three ARNs that profile resolves to.

Nothing here is secret: the region and model id come from environment
variables that the Terraform sets on the runtime (``BEDROCK_MODEL_ID``), with
sensible defaults matching the Claude backend.
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

# Defaults mirror the Terraform contract (variables.tf). NOTE: the model id
# was corrected to Claude Opus 5 via the global CRIS inference profile; the
# Terraform bedrock_model_id default must be updated to match (that file is a
# separate work item — flagged in this directory's README).
#   bedrock_model_id -> global.anthropic.claude-opus-5
#   aws_region       default = us-east-1
DEFAULT_MODEL_ID = "global.anthropic.claude-opus-5"
DEFAULT_REGION = "us-east-1"
ANTHROPIC_VERSION = "bedrock-2023-05-31"


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


class BedrockClient:
    """Thin wrapper around ``bedrock-runtime`` for Claude Messages calls."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or get_model_id()
        self.region_name = region_name or get_region()
        # Retries help with transient Bedrock throttling during a multi
        # sub-task run.
        self._client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region_name,
            config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
        )
        logger.info(
            "BedrockClient ready (model=%s region=%s)",
            self.model_id,
            self.region_name,
        )

    def invoke(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.2,
    ) -> str:
        """Invoke the model with a single user prompt and return the text.

        Args:
            prompt: The user message content.
            system: Optional system prompt.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The concatenated text of the model's response content blocks.

        Raises:
            ClientError: On an AWS/Bedrock API error (surfaced to the caller so
                the agent can fail the run rather than silently continue).
        """
        body: dict = {
            "anthropic_version": ANTHROPIC_VERSION,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system

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
        return self._extract_text(payload)

    @staticmethod
    def _extract_text(payload: dict) -> str:
        """Pull the text out of a Claude Messages API response payload."""
        blocks = payload.get("content") or []
        parts = [b.get("text", "") for b in blocks if b.get("type", "text") == "text"]
        text = "".join(parts).strip()
        if not text:
            raise ValueError("Bedrock response contained no text content")
        return text

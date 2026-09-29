"""Dispatcher Lambda for the AgentCore decomposer backend.

Receives an event shaped like::

    {
        "issue_number": 42,
        "title": "agent-...",
        "body": "...",
        "base_branch": "main",
        "repo": "kaushalavardhanam/kaushalavardhanam"
    }

validates it, and invokes the Bedrock AgentCore Runtime agent (ARN in the
``AGENT_RUNTIME_ARN`` env var, set by Terraform). The event is passed through
verbatim as the invocation payload: the runtime's ``process_invocation`` (see
``../../agent/agent.py``) expects exactly this dict shape
``{issue_number, title, body, base_branch, repo}``.

Dependency-light: boto3 ships in the Lambda Python runtime by default, so no
extra packaging is required.
"""

from __future__ import annotations

import json
import os

import boto3
from botocore.exceptions import BotoCoreError, ClientError

REQUIRED_FIELDS = ("issue_number", "title", "body", "base_branch", "repo")


def _read_response_body(resp):
    """Read the invoke_agent_runtime response into a JSON-safe object.

    The data-plane ``invoke_agent_runtime`` returns the agent output under the
    ``response`` key as a botocore ``StreamingBody`` (NOT ``payload`` — that is
    the request field). Read it fully and try to JSON-decode it; fall back to
    the raw text if it is not JSON.
    """
    body = resp.get("response")
    if body is None:
        return None
    raw = body.read() if hasattr(body, "read") else body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        return raw


def handler(event, context):  # noqa: ANN001 - Lambda signature
    event = event or {}
    agent_runtime_arn = os.environ.get("AGENT_RUNTIME_ARN", "")

    missing = [f for f in REQUIRED_FIELDS if f not in event]
    if missing:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "missing fields", "missing": missing}),
        }

    if not agent_runtime_arn:
        return {
            "statusCode": 502,
            "body": json.dumps({"error": "AGENT_RUNTIME_ARN is not configured"}),
        }

    client = boto3.client("bedrock-agentcore")

    # Pass the validated event through unchanged as the runtime payload — it is
    # the exact shape process_invocation() consumes.
    payload = json.dumps(event).encode("utf-8")

    try:
        resp = client.invoke_agent_runtime(
            agentRuntimeArn=agent_runtime_arn,
            payload=payload,
            contentType="application/json",
            accept="application/json",
        )
    except (ClientError, BotoCoreError) as exc:
        # Botocore/client failure invoking the runtime -> 502 upstream error.
        return {
            "statusCode": 502,
            "body": json.dumps(
                {
                    "error": "invoke_agent_runtime failed",
                    "detail": str(exc),
                    "agent_runtime_arn": agent_runtime_arn,
                }
            ),
        }

    result = _read_response_body(resp)

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "invoked AgentCore Runtime",
                "agent_runtime_arn": agent_runtime_arn,
                "issue_number": event["issue_number"],
                "repo": event["repo"],
                "base_branch": event["base_branch"],
                "runtime_session_id": resp.get("runtimeSessionId"),
                "runtime_status_code": resp.get("statusCode"),
                "runtime_response": result,
            }
        ),
    }

"""Dispatcher Lambda for the AgentCore coding agent.

Invoked asynchronously by CI (``.github/scripts/kick_agentcore_agent.py`` and
``kick_agentcore_fix.py``) with one of::

    {"mode": "issue", "issue_number": 42, "title": "agent-...", "body": "...",
     "base_branch": "main", "repo": "owner/name"}
    {"mode": "fix", "repo": "owner/name", "pr_number": 25, "comment_id": 123}

(``mode`` defaults to ``issue``.) It validates the event and invokes the
Bedrock AgentCore Runtime (``AGENT_RUNTIME_ARN``) with:

* a DETERMINISTIC ``runtimeSessionId`` — one per issue, one per fix comment —
  so a duplicate delivery reaches the same session (which answers
  ``already_running``) instead of launching a second microVM;
* the event plus ``runtime_arn``, so the runtime can stop its own session as
  soon as the job finishes.

The runtime acknowledges within seconds and does the work in the background,
so this function returns well inside its timeout. Async retries are disabled
in Terraform (``aws_lambda_function_event_invoke_config``).

Dependency-light: boto3 ships in the Lambda Python runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

import boto3
from botocore.exceptions import BotoCoreError, ClientError

REQUIRED_FIELDS = {
    "issue": ("issue_number", "title", "body", "base_branch", "repo"),
    "fix": ("repo", "pr_number", "comment_id"),
}
TITLE_PREFIX = "agent-"


def session_id_for(event: dict) -> str:
    """One runtime session per unit of work (AgentCore requires >= 33 chars)."""
    if event.get("mode") == "fix":
        key = f"{event['repo']}-pr-{event['pr_number']}-comment-{event['comment_id']}"
    else:
        key = f"{event['repo']}-issue-{event['issue_number']}"
    readable = re.sub(r"[^A-Za-z0-9-]", "-", key)[:200]
    return f"{readable}-{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def _response(status: int, **body) -> dict:
    return {"statusCode": status, "body": json.dumps(body)}


def _read_response_body(resp):
    """The runtime's reply arrives under ``response`` as a StreamingBody."""
    body = resp.get("response")
    if body is None:
        return None
    raw = body.read() if hasattr(body, "read") else body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(raw) if raw else None
    except (ValueError, TypeError):
        return raw


def handler(event, context):  # noqa: ANN001 - Lambda signature
    event = dict(event or {})
    event.setdefault("mode", "issue")
    required = REQUIRED_FIELDS.get(event["mode"])
    if required is None:
        return _response(400, error=f"unknown mode {event['mode']!r}")
    missing = [f for f in required if f not in event]
    if missing:
        return _response(400, error="missing fields", missing=missing)
    # Launch gate: only agent-* issues ever start a session from here.
    if event["mode"] == "issue" and not str(event["title"]).lower().startswith(TITLE_PREFIX):
        return _response(400, error=f"title does not start with {TITLE_PREFIX!r}")

    agent_runtime_arn = os.environ.get("AGENT_RUNTIME_ARN", "")
    if not agent_runtime_arn:
        return _response(502, error="AGENT_RUNTIME_ARN is not configured")

    session_id = session_id_for(event)
    payload = json.dumps({**event, "runtime_arn": agent_runtime_arn}).encode("utf-8")
    try:
        resp = boto3.client("bedrock-agentcore").invoke_agent_runtime(
            agentRuntimeArn=agent_runtime_arn,
            runtimeSessionId=session_id,
            payload=payload,
            contentType="application/json",
            accept="application/json",
        )
    except (ClientError, BotoCoreError) as exc:
        return _response(502, error="invoke_agent_runtime failed", detail=str(exc),
                         agent_runtime_arn=agent_runtime_arn)

    return _response(
        200,
        message="invoked AgentCore Runtime",
        mode=event["mode"],
        issue_number=event.get("issue_number"),
        pr_number=event.get("pr_number"),
        repo=event["repo"],
        runtime_session_id=resp.get("runtimeSessionId") or session_id,
        runtime_response=_read_response_body(resp),
    )

"""Dispatcher Lambda for the AgentCore decomposer backend.

MINIMAL STUB — the real handler is a separate work item.

Receives an event shaped like::

    {
        "issue_number": 42,
        "title": "agent-...",
        "body": "...",
        "base_branch": "main",
        "repo": "kaushalavardhanam/kaushalavardhanam"
    }

and invokes the Bedrock AgentCore Runtime agent (ARN in AGENT_RUNTIME_ARN).
For now it validates the payload and returns the invocation parameters it
*would* send, so the wiring (OIDC role -> Lambda -> AgentCore) can be
exercised before the runtime container exists.
"""

from __future__ import annotations

import json
import os

REQUIRED_FIELDS = ("issue_number", "title", "body", "base_branch", "repo")


def handler(event, context):  # noqa: ANN001 - Lambda signature
    agent_runtime_arn = os.environ.get("AGENT_RUNTIME_ARN", "")

    missing = [f for f in REQUIRED_FIELDS if f not in (event or {})]
    if missing:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "missing fields", "missing": missing}),
        }

    # Real implementation (later work item) will call:
    #   client = boto3.client("bedrock-agentcore")
    #   client.invoke_agent_runtime(
    #       agentRuntimeArn=agent_runtime_arn,
    #       payload=json.dumps(event).encode("utf-8"),
    #   )
    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": "stub: would invoke AgentCore Runtime",
                "agent_runtime_arn": agent_runtime_arn,
                "issue_number": event["issue_number"],
                "repo": event["repo"],
                "base_branch": event["base_branch"],
            }
        ),
    }

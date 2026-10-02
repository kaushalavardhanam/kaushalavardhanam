"""AgentCore Runtime entrypoint for the Claude coding agent.

A session exists only because the dispatcher Lambda invoked one, which happens
only for an ``agent-*`` issue moved to In Progress or an authorized
``/agent fix`` comment on an open agent PR. Per invocation:

1. validate the payload (``mode`` ``issue`` | ``fix``);
2. de-duplicate: the dispatcher uses ONE runtime session id per issue / per
   fix comment, so a replayed invocation lands in the same session and is
   answered ``already_running`` instead of starting a second job;
3. register an async task (``/ping`` reports ``HealthyBusy``, which keeps the
   session alive past the 15-minute idle timeout), start the job on a
   background thread, and RETURN IMMEDIATELY with ``accepted`` — so the
   dispatcher finishes in seconds rather than outliving its timeout;
4. when the job ends (success or failure), complete the task and stop this
   runtime session (``StopRuntimeSession``) instead of idling for 15 minutes.

Payloads (from ``../terraform/lambda/dispatcher.py``)::

    {"mode": "issue", "issue_number": 12, "title": "agent-...", "body": "...",
     "base_branch": "main", "repo": "owner/name", "runtime_arn": "..."}
    {"mode": "fix", "repo": "owner/name", "pr_number": 25, "comment_id": 123,
     "runtime_arn": "..."}

``mode`` defaults to ``issue`` for payloads from older dispatchers.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Dict, Optional

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("agentcore.agent")

REQUIRED = {
    "issue": ("issue_number", "title", "base_branch", "repo"),
    "fix": ("repo", "pr_number", "comment_id"),
}

_lock = threading.Lock()
_running: set = set()


def job_key(payload: Dict[str, Any]) -> str:
    if payload.get("mode") == "fix":
        return f"fix:{payload['repo']}#{payload['pr_number']}:{payload['comment_id']}"
    return f"issue:{payload['repo']}#{payload['issue_number']}"


def validate(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload or {})
    payload.setdefault("mode", "issue")
    if not payload.get("repo"):
        payload["repo"] = os.environ.get("GITHUB_REPO", "")
    required = REQUIRED.get(payload["mode"])
    if required is None:
        raise ValueError(f"unknown mode {payload['mode']!r}")
    missing = [f for f in required if payload.get(f) in (None, "")]
    if missing:
        raise ValueError(f"invocation payload missing fields: {missing}")
    return payload


def run_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run one job synchronously (the background thread's body)."""
    import jobs
    from github_app import get_installation_token

    if payload["mode"] == "fix":
        return jobs.run_fix_job(payload, get_token=get_installation_token)
    return jobs.run_issue_job(payload, get_token=get_installation_token)


def stop_session(runtime_arn: Optional[str], session_id: Optional[str]) -> None:
    """End this runtime session now rather than waiting out the idle timeout."""
    if not (runtime_arn and session_id):
        logger.info("No runtime ARN/session id; leaving the session to the idle timeout")
        return
    import boto3

    try:
        boto3.client("bedrock-agentcore").stop_runtime_session(
            agentRuntimeArn=runtime_arn, runtimeSessionId=session_id)
        logger.info("Stopped runtime session %s", session_id)
    except Exception:  # noqa: BLE001 - the idle timeout is the fallback
        logger.exception("StopRuntimeSession failed; the idle timeout will reclaim the session")


def handle(payload: Dict[str, Any], session_id: Optional[str], *,
           add_task: Callable[[str], Any] = lambda name: None,
           complete_task: Callable[[Any], Any] = lambda task: None,
           job: Callable[[Dict[str, Any]], Dict[str, Any]] = run_job,
           stop: Callable[[Optional[str], Optional[str]], None] = stop_session,
           background: bool = True) -> Dict[str, Any]:
    """Accept one invocation; the job itself runs on a background thread."""
    payload = validate(payload)
    key = job_key(payload)
    with _lock:
        if key in _running:
            logger.info("Job %s already running in this session; ignoring duplicate", key)
            return {"status": "already_running", "job": key}
        _running.add(key)
    task = add_task(key)

    def body() -> None:
        try:
            result = job(payload)
            logger.info("Job %s finished: %s", key, result)
        except Exception:  # noqa: BLE001 - jobs report failures on GitHub themselves
            logger.exception("Job %s failed", key)
        finally:
            with _lock:
                _running.discard(key)
                idle = not _running
            complete_task(task)
            if idle:
                stop(payload.get("runtime_arn"), session_id)

    if background:
        threading.Thread(target=body, name=key, daemon=True).start()
    else:
        body()
    logger.info("Accepted %s (session %s)", key, session_id)
    return {"status": "accepted", "job": key}


# --------------------------------------------------------------------------- #
# Serving (the AgentCore SDK serves /invocations + /ping on :8080)
# --------------------------------------------------------------------------- #

def main() -> None:
    from bedrock_agentcore.runtime import BedrockAgentCoreApp

    app = BedrockAgentCoreApp()

    @app.entrypoint
    def invoke(payload, context):  # noqa: ANN001 - SDK signature
        try:
            return handle(payload, getattr(context, "session_id", None),
                          add_task=lambda name: app.add_async_task(name),
                          complete_task=app.complete_async_task)
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}

    app.run()


if __name__ == "__main__":
    main()

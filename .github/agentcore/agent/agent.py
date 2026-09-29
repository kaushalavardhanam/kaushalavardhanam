"""AgentCore Runtime entrypoint for the decomposer agent.

The Bedrock AgentCore Runtime invokes a containerized agent over HTTP: the
container must serve ``POST /invocations`` (the invocation payload as the JSON
body) and ``GET /ping`` (health check) on port 8080. See the Terraform
``aws_bedrockagentcore_agent_runtime`` resource, whose ``container_uri`` points
at the image built from this directory.

This module exposes a single ``process_invocation(payload)`` function holding
all the business logic, and serves it two ways:

* If the ``bedrock_agentcore`` SDK is installed (it is, via requirements.txt),
  the ``BedrockAgentCoreApp`` entrypoint is used — the canonical path.
* Otherwise a tiny stdlib ``http.server`` implements the same ``/invocations``
  + ``/ping`` contract, so the container is self-contained and the logic is
  testable without the SDK.

Invocation payload (matches the Terraform dispatcher, ``lambda/dispatcher.py``)::

    {"issue_number": 42, "title": "...", "body": "...",
     "base_branch": "main", "repo": "owner/name"}
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from github_app import get_installation_token
from git_pr import open_pull_request

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("agentcore.decomposer")

# Two decomposer implementations are available side by side so either can be
# demoed. ``DECOMPOSER_IMPL`` selects between them; it DEFAULTS to ``langgraph``
# so existing behaviour is preserved unless explicitly overridden.
#
# * ``langgraph`` (default) — the hand-rolled sentinel-contract workflow in
#   ``decomposer.py``.
# * ``strands`` — the Strands Agents SDK workflow in ``decomposer_strands.py``,
#   which uses native structured output (schema-validated file objects) instead
#   of the JSON/sentinel parsing.
#
# Both expose the same ``run_decomposition(issue_number, title, body, repo,
# base_branch) -> state`` contract, so nothing downstream changes.
DECOMPOSER_IMPL = os.environ.get("DECOMPOSER_IMPL", "langgraph").strip().lower()


def _select_run_decomposition():
    """Return the ``run_decomposition`` callable for the configured impl.

    Imported lazily so selecting one implementation never requires the other's
    dependencies to be installed (e.g. running with the default langgraph impl
    does not import ``strands``, and vice versa).
    """
    if DECOMPOSER_IMPL == "strands":
        logger.info("Using Strands decomposer (DECOMPOSER_IMPL=strands)")
        from decomposer_strands import run_decomposition as _run
        return _run
    if DECOMPOSER_IMPL not in ("langgraph", ""):
        logger.warning(
            "Unknown DECOMPOSER_IMPL=%r; falling back to 'langgraph'.",
            DECOMPOSER_IMPL,
        )
    logger.info("Using LangGraph decomposer (DECOMPOSER_IMPL=langgraph)")
    from decomposer import run_decomposition as _run
    return _run

# The invocation payload contract from the Terraform dispatcher
# (../terraform/lambda/dispatcher.py). ``body`` may be empty; the rest must be
# present and non-empty (``repo`` may be backstopped by the GITHUB_REPO env).
REQUIRED_FIELDS = ("issue_number", "title", "body", "base_branch", "repo")


def process_invocation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full decompose -> implement -> PR pipeline for one issue.

    Validates ``payload`` against :data:`REQUIRED_FIELDS`, then returns a
    JSON-serialisable dict describing the outcome (branch, commit, PR
    url/number, sub-task count). Raises on hard failures so the runtime
    records the invocation as failed.
    """
    payload = payload or {}

    # body may legitimately be empty, so only require its KEY to be present;
    # all other fields must be present and non-empty. repo can be backstopped
    # by the GITHUB_REPO env the Terraform sets on the runtime.
    if "body" not in payload:
        raise ValueError("invocation payload missing field: body")
    repo = str(payload.get("repo") or os.environ.get("GITHUB_REPO", ""))
    non_empty_required = ("issue_number", "title", "base_branch")
    missing = [f for f in non_empty_required if payload.get(f) in (None, "")]
    if not repo:
        missing.append("repo")
    if missing:
        raise ValueError(f"invocation payload missing fields: {missing}")

    base_branch = str(payload.get("base_branch") or "main")
    issue_number = int(payload["issue_number"])
    title = str(payload["title"])
    body = str(payload.get("body") or "")

    logger.info("Processing issue #%s in %s (base=%s)", issue_number, repo, base_branch)

    # 1 + 2 + 3: decompose and implement via Bedrock, using the configured
    # decomposer implementation (langgraph by default, strands when selected).
    run_decomposition = _select_run_decomposition()
    state = run_decomposition(
        issue_number=issue_number,
        title=title,
        body=body,
        repo=repo,
        base_branch=base_branch,
    )
    files = state.get("files", {})
    notes = state.get("implementation_notes", [])
    logger.info("Generated %d file(s) across %d sub-task(s)", len(files), len(state.get("subtasks", [])))

    # 4: authenticate as the GitHub App and open the PR.
    token = get_installation_token()
    pr = open_pull_request(
        repo=repo,
        base_branch=base_branch,
        issue_number=issue_number,
        issue_title=title,
        files=files,
        notes=notes,
        token=token,
    )

    result = {
        "status": "ok",
        "issue_number": issue_number,
        "repo": repo,
        "base_branch": base_branch,
        "branch": pr.branch,
        "commit_sha": pr.commit_sha,
        "pr_url": pr.pr_url,
        "pr_number": pr.pr_number,
        "subtasks": len(state.get("subtasks", [])),
        "files_changed": sorted(files),
    }
    logger.info("Done: PR %s (%s)", pr.pr_number, pr.pr_url)
    return result


# --------------------------------------------------------------------------- #
# Serving
# --------------------------------------------------------------------------- #

try:  # Canonical path: the AgentCore SDK.
    from bedrock_agentcore.runtime import BedrockAgentCoreApp  # type: ignore

    app = BedrockAgentCoreApp()

    @app.entrypoint
    def invoke(payload):  # noqa: ANN001 - SDK signature
        """AgentCore entrypoint: delegates to ``process_invocation``."""
        return process_invocation(payload)

    def main() -> None:
        # The SDK serves /invocations + /ping on 0.0.0.0:8080.
        app.run()

except ImportError:  # Fallback: stdlib HTTP server implementing the contract.
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    app = None  # type: ignore

    class _Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: Dict[str, Any]) -> None:
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - stdlib signature
            if self.path.rstrip("/") == "/ping":
                self._send(200, {"status": "healthy"})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802 - stdlib signature
            if self.path.rstrip("/") != "/invocations":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw or b"{}")
                result = process_invocation(payload)
                self._send(200, result)
            except ValueError as exc:
                self._send(400, {"status": "error", "error": str(exc)})
            except Exception as exc:  # noqa: BLE001 - report and 500
                logger.exception("Invocation failed")
                self._send(500, {"status": "error", "error": str(exc)})

        def log_message(self, *args):  # silence default stderr access log
            return

    def main() -> None:
        port = int(os.environ.get("PORT", "8080"))
        server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
        logger.info("Serving /invocations + /ping on 0.0.0.0:%d", port)
        server.serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Dispatch a reviewer's ``/agent fix`` PR comment to the AgentCore backend.

Triggered by ``issue_comment: created`` (see
``.github/workflows/agentcore-agent-pr-feedback.yml``). When someone comments on
an agent-opened PR (head branch ``agentcore/*``) with::

    /agent fix
    pytest tests/test_vocabulary.py fails at import: use ...

this script checks the commenter is allowed to steer the agent, acknowledges
the comment, and asynchronously invokes the dispatcher Lambda with::

    {"mode": "fix", "repo", "pr_number", "comment_id"}

The AgentCore runtime (``.github/agentcore/agent/pr_fixer.py``) then reads the
comment, the PR, the linked issue and the relevant source, pushes a fix commit
to the same PR branch, and replies on the PR.

Who may trigger a fix: the author of the issue the PR closes, or anyone with
write access to the repo. Everyone else is ignored (with a reply) — the comment
text becomes model input and leads to a push, so it must come from a trusted
person.
"""

from __future__ import annotations

import os
import re
import sys

from kick_agentcore_agent import GITHUB_API, env, github_request, invoke_dispatcher

FIX_COMMAND = "/agent fix"
AGENT_BRANCH_PREFIX = "agentcore/"
ACK_MARKER = "<!-- agentcore-fix-kickoff:{comment_id} -->"
WRITE_PERMISSIONS = {"admin", "maintain", "write"}

_CLOSES_RE = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)", re.IGNORECASE)
_BRANCH_ISSUE_RE = re.compile(r"-(\d+)$")


def is_fix_command(body: str) -> bool:
    first = (body or "").strip().splitlines()[0:1]
    return bool(first) and first[0].strip().lower().startswith(FIX_COMMAND)


def linked_issue_number(pr_body: str, head_branch: str) -> int | None:
    """Mirror of pr_fixer.linked_issue_number (kept dependency-free for CI)."""
    m = _CLOSES_RE.search(pr_body or "")
    if m:
        return int(m.group(1))
    m = _BRANCH_ISSUE_RE.search(head_branch or "")
    return int(m.group(1)) if m else None


def is_authorized(repo: str, user: str, pr: dict, token: str) -> tuple[bool, str]:
    """Issue author, or a collaborator with write access. Returns (ok, reason)."""
    issue_number = linked_issue_number(pr.get("body") or "", pr["head"]["ref"])
    if issue_number:
        issue = github_request(f"{GITHUB_API}/repos/{repo}/issues/{issue_number}", token)
        if (issue.get("user") or {}).get("login", "").lower() == user.lower():
            return True, f"author of #{issue_number}"
    try:
        perm = github_request(
            f"{GITHUB_API}/repos/{repo}/collaborators/{user}/permission", token
        )
    except RuntimeError:
        perm = {}
    level = perm.get("permission", "none")
    if level in WRITE_PERMISSIONS:
        return True, f"{level} access"
    return False, f"not the issue author and has {level!r} access"


def already_dispatched(repo: str, pr_number: int, comment_id: int, token: str) -> bool:
    marker = ACK_MARKER.format(comment_id=comment_id)
    comments = github_request(
        f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments?per_page=100", token
    )
    return isinstance(comments, list) and any(marker in (c.get("body") or "") for c in comments)


def comment(repo: str, number: int, body: str, token: str) -> None:
    github_request(f"{GITHUB_API}/repos/{repo}/issues/{number}/comments", token, {"body": body})


def react(repo: str, comment_id: int, reaction: str, token: str) -> None:
    try:
        github_request(
            f"{GITHUB_API}/repos/{repo}/issues/comments/{comment_id}/reactions",
            token,
            {"content": reaction},
        )
    except RuntimeError as exc:  # cosmetic; never block the dispatch on it
        print(f"warning: could not react to comment {comment_id}: {exc}")


def main() -> int:
    repo = env("GITHUB_REPOSITORY")
    token = env("GITHUB_TOKEN")
    function_name = env("DISPATCHER_LAMBDA_NAME", "agentcore-decomposer-dispatcher")
    region = env("AWS_REGION", "us-east-1")
    pr_number = int(env("PR_NUMBER") or 0)
    comment_id = int(env("COMMENT_ID") or 0)
    user = env("COMMENT_AUTHOR")
    body = os.environ.get("COMMENT_BODY", "")

    if not (repo and token and pr_number and comment_id and user):
        print("GITHUB_REPOSITORY, GITHUB_TOKEN, PR_NUMBER, COMMENT_ID, COMMENT_AUTHOR are required",
              file=sys.stderr)
        return 1
    if not is_fix_command(body):
        print(f"ignore comment {comment_id}: not a {FIX_COMMAND!r} command")
        return 0

    pr = github_request(f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}", token)
    head = pr["head"]
    if pr.get("state") != "open":
        print(f"skip PR #{pr_number}: not open")
        return 0
    if not head["ref"].startswith(AGENT_BRANCH_PREFIX) or (head.get("repo") or {}).get("full_name") != repo:
        print(f"skip PR #{pr_number}: {head['ref']!r} is not an in-repo agent branch")
        return 0

    ok, reason = is_authorized(repo, user, pr, token)
    if not ok:
        print(f"reject comment {comment_id} from @{user}: {reason}")
        comment(repo, pr_number,
                f"@{user} `{FIX_COMMAND}` can only be used by the linked issue's author "
                f"or a collaborator with write access.", token)
        return 0
    if already_dispatched(repo, pr_number, comment_id, token):
        print(f"skip comment {comment_id}: already dispatched")
        return 0

    payload = {"mode": "fix", "repo": repo, "pr_number": pr_number, "comment_id": comment_id}
    print(f"dispatching fix for PR #{pr_number} comment {comment_id} from @{user} ({reason})")
    # Claim first (the marker is the dedup key), withdraw it if the dispatch fails.
    claim = github_request(
        f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments", token,
        {"body": f"{ACK_MARKER.format(comment_id=comment_id)}\n"
                 f"On it, @{user} — the AgentCore agent will plan the fix, test it, push to "
                 f"`{head['ref']}`, and reply here when done."},
    )
    try:
        invoke_dispatcher(function_name, region, payload)
    except Exception:
        if isinstance(claim, dict) and claim.get("id"):
            github_request(f"{GITHUB_API}/repos/{repo}/issues/comments/{claim['id']}", token,
                           method="DELETE")
        raise
    react(repo, comment_id, "eyes", token)
    return 0


if __name__ == "__main__":
    sys.exit(main())

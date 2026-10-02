#!/usr/bin/env python3
"""Dispatch a reviewer's ``/agent fix`` PR comment to the AgentCore backend.

Run by ``.github/workflows/agentcore-agent-pr-feedback.yml`` in two modes:

* **One comment** (``issue_comment: created``; ``COMMENT_ID`` set): handle just
  that comment, so a fix starts within seconds.
* **Scan** (``schedule`` / ``workflow_dispatch``; no ``COMMENT_ID``): walk every
  open agent PR (or only ``PR_NUMBER``) and handle each ``/agent fix`` comment
  that has not been answered yet. This catches comments whose event run was
  dropped or failed.

When someone comments on an agent-opened PR (head branch ``agentcore/*``) with::

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

Every handled comment gets a reply carrying a per-comment marker (accepted or
rejected); the markers are the dedup key, so the scan never answers the same
comment twice.
"""

from __future__ import annotations

import os
import re
import sys

from kick_agentcore_agent import GITHUB_API, env, github_request, invoke_dispatcher

FIX_COMMAND = "/agent fix"
AGENT_BRANCH_PREFIX = "agentcore/"
ACK_MARKER = "<!-- agentcore-fix-kickoff:{comment_id} -->"
REJECT_MARKER = "<!-- agentcore-fix-rejected:{comment_id} -->"
WRITE_PERMISSIONS = {"admin", "maintain", "write"}

_CLOSES_RE = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)", re.IGNORECASE)
_BRANCH_ISSUE_RE = re.compile(r"-(\d+)$")
_HANDLED_RE = re.compile(r"<!-- agentcore-fix-(?:kickoff|rejected):(\d+) -->")
PAGE_SIZE = 100


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


def paged(url: str, token: str) -> list[dict]:
    """GET every page of a GitHub list endpoint (``url`` must already have a query)."""
    items: list[dict] = []
    page = 1
    while True:
        batch = github_request(f"{url}&per_page={PAGE_SIZE}&page={page}", token)
        if not isinstance(batch, list):
            break
        items.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        page += 1
    return items


def list_comments(repo: str, pr_number: int, token: str) -> list[dict]:
    return paged(f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments?sort=created", token)


def handled_comment_ids(comments: list[dict]) -> set[int]:
    """Ids of the ``/agent fix`` comments already answered (accepted or rejected)."""
    return {int(m.group(1)) for c in comments for m in _HANDLED_RE.finditer(c.get("body") or "")}


def is_agent_pr(pr: dict, repo: str) -> bool:
    head = pr["head"]
    return head["ref"].startswith(AGENT_BRANCH_PREFIX) and (head.get("repo") or {}).get("full_name") == repo


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


def handle_fix_comment(repo: str, token: str, function_name: str, region: str, pr: dict,
                       comment_id: int, user: str, body: str, handled: set[int]) -> bool:
    """Dispatch one ``/agent fix`` comment. Returns True if the agent was started."""
    pr_number = pr["number"]
    if not is_fix_command(body):
        print(f"ignore comment {comment_id}: not a {FIX_COMMAND!r} command")
        return False
    if pr.get("state") != "open":
        print(f"skip PR #{pr_number}: not open")
        return False
    if not is_agent_pr(pr, repo):
        print(f"skip PR #{pr_number}: {pr['head']['ref']!r} is not an in-repo agent branch")
        return False
    if comment_id in handled:
        print(f"skip comment {comment_id}: already handled")
        return False

    ok, reason = is_authorized(repo, user, pr, token)
    if not ok:
        print(f"reject comment {comment_id} from @{user}: {reason}")
        comment(repo, pr_number,
                f"{REJECT_MARKER.format(comment_id=comment_id)}\n"
                f"@{user} `{FIX_COMMAND}` can only be used by the linked issue's author "
                f"or a collaborator with write access.", token)
        return False

    payload = {"mode": "fix", "repo": repo, "pr_number": pr_number, "comment_id": comment_id}
    print(f"dispatching fix for PR #{pr_number} comment {comment_id} from @{user} ({reason})")
    # Claim first (the marker is the dedup key), withdraw it if the dispatch fails.
    claim = github_request(
        f"{GITHUB_API}/repos/{repo}/issues/{pr_number}/comments", token,
        {"body": f"{ACK_MARKER.format(comment_id=comment_id)}\n"
                 f"On it, @{user} — the AgentCore agent will plan the fix, test it, push to "
                 f"`{pr['head']['ref']}`, and reply here when done."},
    )
    try:
        invoke_dispatcher(function_name, region, payload)
    except Exception:
        if isinstance(claim, dict) and claim.get("id"):
            github_request(f"{GITHUB_API}/repos/{repo}/issues/comments/{claim['id']}", token,
                           method="DELETE")
        raise
    react(repo, comment_id, "eyes", token)
    return True


def scan_open_prs(repo: str, token: str, function_name: str, region: str,
                  only_pr: int = 0) -> tuple[int, int]:
    """Handle every unanswered ``/agent fix`` comment on open agent PRs.

    Returns ``(dispatched, failed)``. One failed dispatch does not stop the scan;
    its claim is withdrawn, so the next scan retries it.
    """
    if only_pr:
        prs = [github_request(f"{GITHUB_API}/repos/{repo}/pulls/{only_pr}", token)]
    else:
        prs = paged(f"{GITHUB_API}/repos/{repo}/pulls?state=open", token)
    dispatched = failed = 0
    for pr in prs:
        if pr.get("state") != "open" or not is_agent_pr(pr, repo):
            continue
        comments = list_comments(repo, pr["number"], token)
        handled = handled_comment_ids(comments)
        for c in comments:
            user = c.get("user") or {}
            if user.get("type") == "Bot" or c["id"] in handled or not is_fix_command(c.get("body") or ""):
                continue
            try:
                if handle_fix_comment(repo, token, function_name, region, pr, c["id"],
                                      user.get("login", ""), c.get("body") or "", handled):
                    dispatched += 1
            except Exception as exc:
                failed += 1
                print(f"error: PR #{pr['number']} comment {c['id']}: {exc}", file=sys.stderr)
    return dispatched, failed


def main() -> int:
    repo = env("GITHUB_REPOSITORY")
    token = env("GITHUB_TOKEN")
    function_name = env("DISPATCHER_LAMBDA_NAME", "agentcore-decomposer-dispatcher")
    region = env("AWS_REGION", "us-east-1")
    pr_number = int(env("PR_NUMBER") or 0)
    comment_id = int(env("COMMENT_ID") or 0)
    user = env("COMMENT_AUTHOR")
    body = os.environ.get("COMMENT_BODY", "")

    if not (repo and token):
        print("GITHUB_REPOSITORY and GITHUB_TOKEN are required", file=sys.stderr)
        return 1

    if not comment_id:
        scope = f"PR #{pr_number}" if pr_number else "open agent PRs"
        print(f"scanning {scope} for unanswered {FIX_COMMAND!r} comments")
        dispatched, failed = scan_open_prs(repo, token, function_name, region, pr_number)
        print(f"dispatched {dispatched} fix(es), {failed} failed")
        return 1 if failed else 0

    if not (pr_number and user):
        print("PR_NUMBER and COMMENT_AUTHOR are required with COMMENT_ID", file=sys.stderr)
        return 1
    if not is_fix_command(body):
        print(f"ignore comment {comment_id}: not a {FIX_COMMAND!r} command")
        return 0
    pr = github_request(f"{GITHUB_API}/repos/{repo}/pulls/{pr_number}", token)
    handled = handled_comment_ids(list_comments(repo, pr_number, token))
    handle_fix_comment(repo, token, function_name, region, pr, comment_id, user, body, handled)
    return 0


if __name__ == "__main__":
    sys.exit(main())

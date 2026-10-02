"""The two jobs the runtime performs, end to end, around the orchestrator.

* :func:`run_issue_job` — an ``agent-*`` issue -> a new ``agentcore/*`` branch
  and PR (a DRAFT PR if any sub-task or verify command did not pass).
* :func:`run_fix_job` — a reviewer's ``/agent fix`` comment -> new commits on
  the existing PR branch and a reply on the PR.

Both RE-CHECK the launch conditions against GitHub before doing any work (the
issue is open and titled ``agent-*``; the PR is open, in-repo, on an agent
branch), so a stray or replayed invocation exits in seconds without spending
model tokens. Every outcome is reported back on GitHub.

The GitHub token is used only in this process (API calls, clone, push) and is
re-minted for the push because a run can outlive a 1-hour installation token;
the Claude sessions never see it.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Callable, Dict, List, Optional

import git_pr
from claude_runner import ClaudeRunner
from orchestrator import DONE, Ask, OrchestrationError, Orchestrator, Outcome

logger = logging.getLogger(__name__)

TITLE_PREFIX = "agent-"
AGENT_BRANCH_PREFIX = "agentcore/"
FIX_COMMAND = "/agent fix"
FIX_TRAILER = "Agent-Fix-Comment"
RESULT_MARKER = "<!-- agentcore-result -->"

_CLOSES_RE = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)", re.IGNORECASE)
_BRANCH_ISSUE_RE = re.compile(r"-(\d+)$")

TokenProvider = Callable[[], str]
RunnerFactory = Callable[[str], ClaudeRunner]


def strip_command(body: str) -> str:
    """Drop the leading ``/agent fix`` token, keeping the reviewer's details."""
    text = (body or "").strip()
    if text.lower().startswith(FIX_COMMAND):
        text = text[len(FIX_COMMAND):]
    return text.strip()


def linked_issue_number(pr_body: str, head_branch: str) -> Optional[int]:
    """The issue a PR closes: ``Closes #N`` in the body, else the branch suffix."""
    m = _CLOSES_RE.search(pr_body or "")
    if m:
        return int(m.group(1))
    m = _BRANCH_ISSUE_RE.search(head_branch or "")
    return int(m.group(1)) if m else None


def default_runner(clone_dir: str) -> ClaudeRunner:
    test_bin = os.environ.get("TEST_VENV_BIN", "")
    extra = {"PATH": f"{test_bin}:{os.environ.get('PATH', '')}"} if test_bin else {}
    return ClaudeRunner(clone_dir, extra_env=extra)


def _command_env() -> Dict[str, str]:
    test_bin = os.environ.get("TEST_VENV_BIN", "")
    return {"PATH": f"{test_bin}:{os.environ.get('PATH', '')}"} if test_bin else {}


def _budget() -> float:
    return float(os.environ.get("AGENT_BUDGET_USD", "20"))


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

_ICON = {DONE: "✅", "failed": "❌", "skipped": "⏭️"}


def format_report(outcome: Outcome) -> str:
    lines = ["## Requirements"]
    covered: Dict[str, List[str]] = {}
    for rec in outcome.records:
        if rec.status == DONE and rec.depth == 0:
            for rid in rec.task.covers:
                covered.setdefault(rid, []).append(str(rec.task.id))
    for r in outcome.requirements.items:
        by = covered.get(r.id)
        lines.append(f"- {'✅' if by else '❌'} **{r.id}** {r.text}" + (f" — sub-task {', '.join(by)}" if by else ""))

    lines += ["", "## Sub-tasks"]
    for rec in outcome.records:
        indent = "  " * rec.depth
        sha = f" `{rec.commit[:10]}`" if rec.commit else ""
        lines.append(f"{indent}- {_ICON.get(rec.status, '•')} {rec.task.id}. {rec.task.title}{sha}"
                     f" — `{rec.task.acceptance.command}`")
        if rec.detail and rec.status != DONE:
            lines.append(f"{indent}  <details><summary>details</summary>\n\n```\n{rec.detail[-1500:]}\n```\n</details>")
        elif rec.detail:
            lines.append(f"{indent}  {rec.detail.splitlines()[0][:300]}")

    if outcome.verify:
        base = {(b.command.cwd, b.command.command): b.ok for b in outcome.baseline}
        lines += ["", "## Verification", "| command | before | after |", "|---|---|---|"]
        for v in outcome.verify:
            before = base.get((v.command.cwd, v.command.command))
            lines.append(f"| `cd {v.command.cwd} && {v.command.command}` | "
                         f"{'pass' if before else 'fail'} | {'✅ pass' if v.ok else '❌ fail'} |")
        for v in outcome.unresolved:
            lines.append(f"\n<details><summary>{v.command.command} output</summary>\n\n```\n"
                         f"{v.output[-2500:]}\n```\n</details>")
    if outcome.notes:
        lines += ["", "## Notes", *[f"- {n}" for n in outcome.notes]]
    lines += ["", f"_Model spend: ${outcome.cost_usd:.2f}_"]
    return "\n".join(lines)


def _reply(repo: str, number: int, body: str, token: str) -> None:
    git_pr.post_issue_comment(repo, number, f"{RESULT_MARKER}\n{body}", token)


# --------------------------------------------------------------------------- #
# Issue -> PR
# --------------------------------------------------------------------------- #


def run_issue_job(payload: dict, *, get_token: TokenProvider,
                  runner_factory: RunnerFactory = default_runner,
                  workdir: Optional[str] = None) -> dict:
    repo = str(payload["repo"])
    number = int(payload["issue_number"])
    token = get_token()
    issue = git_pr.github_api("GET", f"/repos/{repo}/issues/{number}", token)

    # Launch-condition re-check: never trust the invocation alone.
    title = issue.get("title") or ""
    if issue.get("pull_request") or issue.get("state") != "open" or not title.lower().startswith(TITLE_PREFIX):
        logger.info("Issue #%s no longer qualifies (state=%s title=%r); exiting", number,
                    issue.get("state"), title)
        return {"status": "skipped", "reason": "issue is not an open agent-* issue"}

    base = str(payload.get("base_branch") or "main")
    branch = git_pr.issue_branch(title, number)
    if git_pr.remote_branch_exists(repo, branch, token):
        _reply(repo, number, f"Branch `{branch}` already exists, so I did not start over. "
               "Comment `/agent fix <details>` on its PR to change it.", token)
        return {"status": "skipped", "reason": f"{branch} exists"}

    clone = git_pr.clone_branch(repo=repo, branch=base, token=token, workdir=workdir, depth=1)
    git_pr.create_branch(clone, branch)
    ask = Ask(kind="issue", repo=repo, title=f"#{number} {title}", body=issue.get("body") or "")
    orch = Orchestrator(clone, runner_factory(clone), budget_usd=_budget(), command_env=_command_env())
    try:
        outcome = orch.run(ask)
    except Exception as exc:  # noqa: BLE001 - report on the issue, then re-raise
        logger.exception("Issue #%s run failed", number)
        what = "could not plan this issue" if isinstance(exc, OrchestrationError) else "hit an error"
        _reply(repo, number, f"❌ The agent {what}, so no PR was opened.\n\n```\n{str(exc)[:1500]}\n```",
               get_token())
        raise

    report = format_report(outcome)
    if not outcome.commits:
        _reply(repo, number, f"❌ No sub-task passed its acceptance check, so no PR was opened.\n\n{report}", token)
        return {"status": "failed", "branch": branch, "cost_usd": outcome.cost_usd}

    token = get_token()
    git_pr.push(clone_dir=clone, repo=repo, branch=branch, token=token)
    draft = not outcome.complete
    header = (f"Automated implementation for #{number}."
              + ("\n\n> **Draft:** some sub-tasks or checks did not pass — see below." if draft else ""))
    url, pr_number = git_pr.create_pr(
        repo=repo, head=branch, base=base, title=f"[agentcore] {title}",
        body=f"{header}\n\n{report}\n\nCloses #{number}", draft=draft, token=token)
    return {"status": "ok" if not draft else "partial", "branch": branch, "pr_url": url,
            "pr_number": pr_number, "commits": outcome.commits, "cost_usd": outcome.cost_usd}


# --------------------------------------------------------------------------- #
# "/agent fix" -> commits on the existing PR
# --------------------------------------------------------------------------- #


def run_fix_job(payload: dict, *, get_token: TokenProvider,
                runner_factory: RunnerFactory = default_runner,
                workdir: Optional[str] = None) -> dict:
    repo = str(payload["repo"])
    pr_number = int(payload["pr_number"])
    comment_id = int(payload["comment_id"])
    token = get_token()
    gh = lambda path: git_pr.github_api("GET", path, token)  # noqa: E731

    pr = gh(f"/repos/{repo}/pulls/{pr_number}")
    head = pr["head"]
    if pr.get("state") != "open" or (head.get("repo") or {}).get("full_name") != repo \
            or not head["ref"].startswith(AGENT_BRANCH_PREFIX):
        logger.info("PR #%s no longer qualifies; exiting", pr_number)
        return {"status": "skipped", "reason": "PR is not an open in-repo agent PR"}

    comment = gh(f"/repos/{repo}/issues/comments/{comment_id}")
    review_comments = [
        f"{rc.get('path')}:{rc.get('line') or rc.get('original_line') or '?'} — "
        f"@{(rc.get('user') or {}).get('login')}: {(rc.get('body') or '').strip()}"
        for rc in gh(f"/repos/{repo}/pulls/{pr_number}/comments?per_page=100") or []
        if (rc.get("user") or {}).get("type") != "Bot"
    ]
    changed = [f["filename"] for f in gh(f"/repos/{repo}/pulls/{pr_number}/files?per_page=100") or []
               if f.get("status") != "removed"]
    issue_number = linked_issue_number(pr.get("body") or "", head["ref"])
    issue = gh(f"/repos/{repo}/issues/{issue_number}") if issue_number else {}

    ask = Ask(
        kind="fix", repo=repo,
        title=f"#{issue_number} {issue.get('title') or pr.get('title') or ''}",
        body=issue.get("body") or pr.get("body") or "",
        feedback=strip_command(comment.get("body") or ""),
        feedback_author=(comment.get("user") or {}).get("login", ""),
        review_comments=review_comments, changed_files=changed,
    )
    link = f"https://github.com/{repo}/pull/{pr_number}#issuecomment-{comment_id}"
    trailer = f"{FIX_TRAILER}: {comment_id}"

    try:
        clone = git_pr.clone_branch(repo=repo, branch=head["ref"], token=token, workdir=workdir)
        if git_pr.has_commit_with_trailer(clone, trailer):
            return {"status": "already_applied", "branch": head["ref"]}
        orch = Orchestrator(clone, runner_factory(clone), budget_usd=_budget(),
                            commit_trailers=[trailer], command_env=_command_env())
        outcome = orch.run(ask)
        if outcome.commits:
            git_pr.push(clone_dir=clone, repo=repo, branch=head["ref"], token=get_token())
    except Exception as exc:  # noqa: BLE001 - report on the PR, then re-raise
        logger.exception("Fix for PR #%s failed", pr_number)
        _reply(repo, pr_number, f"❌ Could not apply [this feedback]({link}).\n\n```\n{str(exc)[:1500]}\n```\n"
               "Comment `/agent fix` again to retry.", get_token())
        raise

    report = format_report(outcome)
    if not outcome.commits:
        headline = f"ℹ️ No changes pushed for [this feedback]({link})."
    elif outcome.complete:
        headline = f"🔧 Pushed {len(outcome.commits)} commit(s) to `{head['ref']}` addressing [this feedback]({link})."
    else:
        headline = (f"⚠️ Pushed {len(outcome.commits)} commit(s) to `{head['ref']}` for [this feedback]({link}), "
                    "but some checks still fail — see below.")
    _reply(repo, pr_number, f"{headline}\n\n{report}\n\nComment `/agent fix <details>` if anything still fails.",
           get_token())
    return {"status": "ok" if outcome.complete else "partial", "branch": head["ref"],
            "commits": outcome.commits, "cost_usd": outcome.cost_usd}

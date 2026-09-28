"""Git + pull-request flow for the AgentCore decomposer agent.

Given the generated files (path -> content) and the issue payload, this:

1. Clones the repo over HTTPS using the GitHub App installation token as the
   credential (embedded only in the in-process remote URL, never logged).
2. Creates a feature branch off ``base_branch`` (``agentcore/<slug>``).
3. Writes the generated files, commits them (identity is the App's bot user),
   and pushes the branch.
4. Opens a PR targeting ``base_branch`` with a body containing
   ``Closes #<issue_number>``.

The installation token is short-lived (1 hour) and is passed in by the caller,
which obtains it from ``github_app.get_installation_token()``.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
REQUEST_TIMEOUT = 30

# The App's commit identity. GitHub attributes commits by this bot account when
# the App is named "<app-slug>"; the numeric id is stable for the app-slug[bot]
# user and can be overridden via env if a specific App is used.
GIT_AUTHOR_NAME = os.environ.get("GIT_AUTHOR_NAME", "agentcore-decomposer[bot]")
GIT_AUTHOR_EMAIL = os.environ.get(
    "GIT_AUTHOR_EMAIL", "agentcore-decomposer[bot]@users.noreply.github.com"
)


@dataclass
class PRResult:
    branch: str
    commit_sha: str
    pr_url: Optional[str]
    pr_number: Optional[int]


def slugify(text: str, max_len: int = 40) -> str:
    """Turn an issue title into a branch-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (slug[:max_len].strip("-")) or "issue"


def _run(cmd: List[str], cwd: str, token: Optional[str] = None) -> str:
    """Run a git command, raising with a redacted message on failure.

    ``token`` (if given) is scrubbed from any error output so it never leaks
    into logs or exceptions.
    """
    try:
        out = subprocess.run(
            cmd, cwd=cwd, check=True, capture_output=True, text=True
        )
        return out.stdout.strip()
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        if token:
            stderr = stderr.replace(token, "***")
        # Also scrub any token that slipped into the command echo.
        safe_cmd = [c.replace(token, "***") if token and token in c else c for c in cmd]
        raise RuntimeError(f"git command failed: {' '.join(safe_cmd)}\n{stderr}") from None


def _authed_remote(repo: str, token: str) -> str:
    """Build an HTTPS remote URL with the installation token as credential."""
    # x-access-token is GitHub's documented username for App installation tokens.
    return f"https://x-access-token:{token}@github.com/{repo}.git"


def open_pull_request(
    *,
    repo: str,
    base_branch: str,
    issue_number: int,
    issue_title: str,
    files: Dict[str, str],
    notes: List[str],
    token: str,
    workdir: Optional[str] = None,
) -> PRResult:
    """Clone, branch, commit, push, and open a PR. Returns the PR result.

    Args:
        repo: ``owner/name``.
        base_branch: Branch to base the feature branch on and target the PR at.
        issue_number: The issue this closes.
        issue_title: Used for the branch slug, commit message, and PR title.
        files: repo-relative path -> full content to write.
        notes: Per-sub-task summaries for the PR body.
        token: GitHub App installation token.
        workdir: Optional parent dir for the clone (a temp dir by default).
    """
    branch = f"agentcore/{slugify(issue_title)}-{issue_number}"
    remote = _authed_remote(repo, token)

    parent = workdir or tempfile.mkdtemp(prefix="agentcore-clone-")
    clone_dir = os.path.join(parent, "repo")

    logger.info("Cloning %s (base=%s)", repo, base_branch)
    _run(
        ["git", "clone", "--depth", "1", "--branch", base_branch, remote, clone_dir],
        cwd=parent,
        token=token,
    )

    # Configure commit identity locally (not global).
    _run(["git", "config", "user.name", GIT_AUTHOR_NAME], cwd=clone_dir)
    _run(["git", "config", "user.email", GIT_AUTHOR_EMAIL], cwd=clone_dir)

    _run(["git", "checkout", "-b", branch], cwd=clone_dir, token=token)

    # Write generated files.
    for rel_path, content in files.items():
        abs_path = os.path.join(clone_dir, rel_path)
        os.makedirs(os.path.dirname(abs_path) or clone_dir, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as fh:
            fh.write(content)

    _run(["git", "add", "-A"], cwd=clone_dir, token=token)
    commit_msg = f"{issue_title}\n\nCloses #{issue_number}"
    _run(["git", "commit", "-m", commit_msg], cwd=clone_dir, token=token)
    commit_sha = _run(["git", "rev-parse", "HEAD"], cwd=clone_dir)

    logger.info("Pushing branch %s", branch)
    _run(["git", "push", "-u", "origin", branch], cwd=clone_dir, token=token)

    pr_url, pr_number = _create_pr(
        repo=repo,
        head=branch,
        base=base_branch,
        issue_number=issue_number,
        issue_title=issue_title,
        notes=notes,
        token=token,
    )
    return PRResult(branch=branch, commit_sha=commit_sha, pr_url=pr_url, pr_number=pr_number)


def _create_pr(
    *,
    repo: str,
    head: str,
    base: str,
    issue_number: int,
    issue_title: str,
    notes: List[str],
    token: str,
):
    """Create the PR via the REST API. Returns (url, number)."""
    body_lines = [
        f"Automated implementation for #{issue_number}.",
        "",
        "## Sub-tasks",
        *[f"- {n}" for n in notes],
        "",
        f"Closes #{issue_number}",
    ]
    payload = {
        "title": f"[agentcore] {issue_title}",
        "head": head,
        "base": base,
        "body": "\n".join(body_lines),
        "maintainer_can_modify": True,
    }
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    logger.info("Opening PR %s -> %s", head, base)
    resp = requests.post(
        f"{GITHUB_API}/repos/{repo}/pulls",
        headers=headers,
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"PR creation failed: {resp.status_code} {resp.text[:300]}")
    data = resp.json()
    return data.get("html_url"), data.get("number")

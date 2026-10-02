"""Git + GitHub REST helpers for the AgentCore coding agent.

Token hygiene: the GitHub App installation token is passed in by the caller
(``github_app.get_installation_token()``), used for one git command at a time,
and never persisted. :func:`clone_branch` rewrites ``origin`` to the
token-free URL immediately after cloning, so the checkout the Claude sessions
work in contains no credential; :func:`push` supplies the token on the command
line for that single push.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
REQUEST_TIMEOUT = 30

# The App's commit identity. GitHub attributes commits by this bot account when
# the App is named "<app-slug>"; override via env if a specific App is used.
GIT_AUTHOR_NAME = os.environ.get("GIT_AUTHOR_NAME", "agentcore-decomposer[bot]")
GIT_AUTHOR_EMAIL = os.environ.get(
    "GIT_AUTHOR_EMAIL", "agentcore-decomposer[bot]@users.noreply.github.com"
)


def slugify(text: str, max_len: int = 40) -> str:
    """Turn an issue title into a branch-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return (slug[:max_len].strip("-")) or "issue"


def issue_branch(title: str, issue_number: int) -> str:
    return f"agentcore/{slugify(title)}-{issue_number}"


def _run(cmd: List[str], cwd: str, token: Optional[str] = None) -> str:
    """Run a git command, raising with a redacted message on failure."""
    try:
        out = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        return out.stdout.strip()
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        if token:
            stderr = stderr.replace(token, "***")
        safe_cmd = [c.replace(token, "***") if token and token in c else c for c in cmd]
        raise RuntimeError(f"git command failed: {' '.join(safe_cmd)}\n{stderr}") from None


def _authed_remote(repo: str, token: str) -> str:
    # x-access-token is GitHub's documented username for App installation tokens.
    return f"https://x-access-token:{token}@github.com/{repo}.git"


def _public_remote(repo: str) -> str:
    return f"https://github.com/{repo}.git"


def clone_branch(*, repo: str, branch: str, token: str, workdir: Optional[str] = None,
                 depth: int = 50) -> str:
    """Clone ``branch`` of ``repo``, strip the token from ``origin``, set identity."""
    parent = workdir or tempfile.mkdtemp(prefix="agentcore-clone-")
    clone_dir = os.path.join(parent, "repo")
    logger.info("Cloning %s (branch=%s)", repo, branch)
    _run(["git", "clone", "--depth", str(depth), "--branch", branch,
          _authed_remote(repo, token), clone_dir], cwd=parent, token=token)
    _run(["git", "remote", "set-url", "origin", _public_remote(repo)], cwd=clone_dir)
    _run(["git", "config", "user.name", GIT_AUTHOR_NAME], cwd=clone_dir)
    _run(["git", "config", "user.email", GIT_AUTHOR_EMAIL], cwd=clone_dir)
    return clone_dir


def remote_branch_exists(repo: str, branch: str, token: str) -> bool:
    out = _run(["git", "ls-remote", "--heads", _authed_remote(repo, token), branch],
               cwd=tempfile.gettempdir(), token=token)
    return bool(out.strip())


def create_branch(clone_dir: str, branch: str) -> None:
    _run(["git", "checkout", "-b", branch], cwd=clone_dir)


def head_sha(clone_dir: str) -> str:
    return _run(["git", "rev-parse", "HEAD"], cwd=clone_dir)


def has_commit_with_trailer(clone_dir: str, trailer: str) -> bool:
    """True if any commit in the (shallow) history carries ``trailer`` verbatim."""
    out = _run(["git", "log", "--format=%B", "--fixed-strings", f"--grep={trailer}"],
               cwd=clone_dir)
    return trailer in out


def push(*, clone_dir: str, repo: str, branch: str, token: str) -> None:
    """Push HEAD to ``branch`` (fast-forward only) using the token for this call only."""
    logger.info("Pushing %s to %s", head_sha(clone_dir)[:12], branch)
    _run(["git", "push", _authed_remote(repo, token), f"HEAD:refs/heads/{branch}"],
         cwd=clone_dir, token=token)


def github_api(method: str, path: str, token: str, payload: Optional[dict] = None):
    """Call the GitHub REST API as the App installation; returns parsed JSON."""
    resp = requests.request(
        method,
        f"{GITHUB_API}{path}",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json=payload,
        timeout=REQUEST_TIMEOUT,
    )
    if resp.status_code >= 300:
        raise RuntimeError(f"GitHub {method} {path} failed: {resp.status_code} {resp.text[:300]}")
    return resp.json() if resp.content else {}


def post_issue_comment(repo: str, number: int, body: str, token: str) -> None:
    """Comment on an issue or PR conversation."""
    github_api("POST", f"/repos/{repo}/issues/{number}/comments", token, {"body": body})


def create_pr(*, repo: str, head: str, base: str, title: str, body: str, draft: bool,
              token: str):
    """Open a PR. Returns (url, number)."""
    logger.info("Opening %sPR %s -> %s", "draft " if draft else "", head, base)
    data = github_api("POST", f"/repos/{repo}/pulls", token, {
        "title": title, "head": head, "base": base, "body": body,
        "draft": draft, "maintainer_can_modify": True,
    })
    return data.get("html_url"), data.get("number")

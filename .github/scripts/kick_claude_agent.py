#!/usr/bin/env python3
"""Dispatch the Claude Code agent workflow when an `agent-*` issue moves to
In Progress on https://github.com/orgs/kaushalavardhanam/projects/1

Triggers:
  - GitHub Project (orgs/kaushalavardhanam/projects/1) Status set to In Progress
    for issues whose title starts with 'agent-'
  - Issue labeled 'in-progress'
  - workflow_dispatch with an issue number

This script only enqueues the run: it fires a workflow_dispatch event against
claude-agent-issue.yml, which runs anthropics/claude-code-action and opens the
PR against main.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

GITHUB_API = "https://api.github.com"
GITHUB_GRAPHQL = "https://api.github.com/graphql"
KICKOFF_MARKER = "<!-- claude-code-agent-kickoff -->"
TITLE_PREFIX = "agent-"
BASE_BRANCH_LABEL_PREFIX = "base:"


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def resolve_base_branch(issue: dict, default_ref: str, override: str = "") -> str:
    """Base branch precedence: explicit override > `base:<branch>` issue label > default."""
    if override:
        return override
    for label in issue.get("labels") or []:
        name = label.get("name") or ""
        if name.lower().startswith(BASE_BRANCH_LABEL_PREFIX):
            branch = name[len(BASE_BRANCH_LABEL_PREFIX):].strip()
            if branch:
                return branch
    return default_ref


def github_request(url: str, token: str, payload: dict | None = None, method: str | None = None) -> dict | list:
    data = None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "kaushalavardhanam-claude-agent-kickoff",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method or ("POST" if data else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub {exc.code} {url}: {detail}") from exc


def graphql(token: str, query: str, variables: dict) -> dict:
    body = github_request(GITHUB_GRAPHQL, token, {"query": query, "variables": variables})
    if body.get("errors"):
        raise RuntimeError(f"GraphQL errors: {json.dumps(body['errors'])}")
    return body["data"]


def dispatch_agent_workflow(
    repo: str, token: str, workflow_file: str, ref: str, issue_number: int, base_branch: str
) -> None:
    github_request(
        f"{GITHUB_API}/repos/{repo}/actions/workflows/{workflow_file}/dispatches",
        token,
        {"ref": ref, "inputs": {"issue_number": str(issue_number), "base_branch": base_branch}},
    )


def already_started(repo: str, issue_number: int, token: str) -> bool:
    comments = github_request(
        f"{GITHUB_API}/repos/{repo}/issues/{issue_number}/comments?per_page=100",
        token,
    )
    if not isinstance(comments, list):
        return False
    return any(KICKOFF_MARKER in (c.get("body") or "") for c in comments)


def fetch_issue(repo: str, number: int, token: str) -> dict:
    return github_request(f"{GITHUB_API}/repos/{repo}/issues/{number}", token)  # type: ignore[return-value]


def start_agent(
    issue: dict, repo: str, comment_token: str, workflow_file: str, default_ref: str, base_override: str = ""
) -> None:
    number = issue["number"]
    title = issue.get("title") or ""
    if not title.lower().startswith(TITLE_PREFIX):
        print(f"skip #{number}: title does not start with {TITLE_PREFIX!r}")
        return
    if issue.get("pull_request"):
        print(f"skip #{number}: pull request")
        return
    if already_started(repo, number, comment_token):
        print(f"skip #{number}: already kicked off")
        return

    base_branch = resolve_base_branch(issue, default_ref, base_override)
    print(f"dispatching {workflow_file} for #{number} against base branch {base_branch!r}")
    dispatch_agent_workflow(repo, comment_token, workflow_file, base_branch, number, base_branch)

    actions_url = f"https://github.com/{repo}/actions/workflows/{workflow_file}"
    comment = (
        f"{KICKOFF_MARKER}\n"
        f"Dispatched the Claude Code agent workflow for this issue against `{base_branch}`.\n\n"
        f"- Watch the run: {actions_url}\n"
        f"- It will open a PR against `{base_branch}` when the work is complete.\n"
    )
    github_request(
        f"{GITHUB_API}/repos/{repo}/issues/{number}/comments",
        comment_token,
        {"body": comment},
    )
    print(f"dispatched workflow for #{number}")


def scan_project(
    owner: str,
    number: int,
    repo: str,
    project_token: str,
    comment_token: str,
    workflow_file: str,
    base_ref: str,
    owner_type: str,
    base_override: str = "",
) -> int:
    root = "organization" if owner_type == "organization" else "user"
    query = f"""
    query ($login: String!, $number: Int!, $cursor: String) {{
      {root}(login: $login) {{
        projectV2(number: $number) {{
          items(first: 50, after: $cursor) {{
            pageInfo {{ hasNextPage endCursor }}
            nodes {{
              fieldValues(first: 20) {{
                nodes {{
                  ... on ProjectV2ItemFieldSingleSelectValue {{
                    name
                    field {{ ... on ProjectV2SingleSelectField {{ name }} }}
                  }}
                }}
              }}
              content {{
                __typename
                ... on Issue {{
                  number
                  title
                  url
                  repository {{ nameWithOwner }}
                }}
              }}
            }}
          }}
        }}
      }}
    }}
    """
    started = 0
    cursor = None
    while True:
        data = graphql(project_token, query, {"login": owner, "number": number, "cursor": cursor})
        project = (data.get(root) or {}).get("projectV2")
        if not project:
            raise RuntimeError(
                f"Project {owner_type} {owner}/{number} not found or token lacks project read access"
            )
        items = project["items"]
        for node in items["nodes"]:
            content = node.get("content") or {}
            if content.get("__typename") != "Issue":
                continue
            if (content.get("repository") or {}).get("nameWithOwner") != repo:
                continue
            status = ""
            for fv in (node.get("fieldValues") or {}).get("nodes") or []:
                field = (fv.get("field") or {}).get("name") or ""
                if normalize(field) == "status":
                    status = fv.get("name") or ""
                    break
            if normalize(status) != "inprogress":
                continue
            issue = fetch_issue(repo, content["number"], comment_token)
            before = already_started(repo, content["number"], comment_token)
            start_agent(issue, repo, comment_token, workflow_file, base_ref, base_override)
            if not before and already_started(repo, content["number"], comment_token):
                started += 1
        if not items["pageInfo"]["hasNextPage"]:
            break
        cursor = items["pageInfo"]["endCursor"]
    return started


def main() -> int:
    repo = env("GITHUB_REPOSITORY")
    comment_token = env("GITHUB_TOKEN")
    project_token = env("GH_PROJECT_TOKEN") or comment_token
    workflow_file = env("CLAUDE_WORKFLOW_FILE", "claude-agent-issue.yml")
    base_ref = env("CLAUDE_BASE_REF", "main")
    base_override = env("CLAUDE_BASE_REF_OVERRIDE")
    project_owner = env("PROJECT_OWNER", "kaushalavardhanam")
    project_number = int(env("PROJECT_NUMBER", "1") or "1")
    owner_type = env("PROJECT_OWNER_TYPE", "organization").lower()
    if owner_type not in {"organization", "user"}:
        print(f"PROJECT_OWNER_TYPE must be organization or user, got {owner_type!r}", file=sys.stderr)
        return 1
    event = env("GITHUB_EVENT_NAME") or env("EVENT_NAME")

    if not repo or not comment_token:
        print("GITHUB_TOKEN and GITHUB_REPOSITORY are required", file=sys.stderr)
        return 1

    if event == "issues":
        label = env("LABEL_NAME")
        issue_number = env("ISSUE_NUMBER")
        if normalize(label) not in {"inprogress", "agent", "agentrun"}:
            print(f"ignore label {label!r}")
            return 0
        issue = fetch_issue(repo, int(issue_number), comment_token)
        start_agent(issue, repo, comment_token, workflow_file, base_ref, base_override)
        return 0

    if event == "workflow_dispatch" and env("ISSUE_NUMBER"):
        issue = fetch_issue(repo, int(env("ISSUE_NUMBER")), comment_token)
        start_agent(issue, repo, comment_token, workflow_file, base_ref, base_override)
        return 0

    print(
        f"scanning {owner_type} project {project_owner}/{project_number} "
        f"for In Progress {TITLE_PREFIX}* issues"
    )
    started = scan_project(
        project_owner,
        project_number,
        repo,
        project_token,
        comment_token,
        workflow_file,
        base_ref,
        owner_type,
        base_override,
    )
    print(f"started {started} agent(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

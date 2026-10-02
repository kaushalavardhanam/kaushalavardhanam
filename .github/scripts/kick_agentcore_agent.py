#!/usr/bin/env python3
"""Kick the AgentCore decomposer backend for an `agent-*` issue.

Instead of firing a Cursor Cloud Agent or a `workflow_dispatch`, this script
authenticates to AWS (via the GitHub Actions OIDC role that the Terraform in
`.github/agentcore/terraform/` creates) and invokes the **dispatcher Lambda**
with a payload shaped exactly as the Lambda's handler expects::

    {"issue_number", "title", "body", "base_branch", "repo"}

Triggers (same as the Cursor backend):
  - GitHub Project (orgs/kaushalavardhanam/projects/1) Status set to In Progress
    for issues whose title starts with 'agent-'
  - Issue labeled 'in-progress' / 'agent' / 'agent-run'
  - workflow_dispatch with an issue number (and optional base_branch override)

Base-branch resolution precedence (highest wins):
  1. workflow_dispatch `base_branch` input (BASE_BRANCH_INPUT env)
  2. a `base:<branch>` label on the issue
  3. DEFAULT_BASE_BRANCH env (defaults to 'main')

The dispatcher Lambda then invokes the Bedrock AgentCore Runtime agent, which
decomposes + implements the issue and opens a PR.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request

import boto3

GITHUB_API = "https://api.github.com"
GITHUB_GRAPHQL = "https://api.github.com/graphql"
KICKOFF_MARKER = "<!-- agentcore-decomposer-kickoff -->"
TITLE_PREFIX = "agent-"
BASE_LABEL_PREFIX = "base:"


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def github_request(url: str, token: str, payload: dict | None = None, method: str | None = None) -> dict | list:
    data = None
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "kaushalavardhanam-agentcore-agent-kickoff",
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


def invoke_dispatcher(function_name: str, region: str, payload: dict) -> dict:
    """Invoke the dispatcher Lambda asynchronously (fire-and-forget).

    The dispatcher fans out to the Bedrock AgentCore Runtime agent, whose full
    decompose -> implement -> open-PR run takes many minutes. A synchronous
    ``RequestResponse`` invoke made this CI step block on that entire run and
    time out at the Lambda client's default 60s read timeout
    (``ReadTimeoutError`` on the ``/functions/.../invocations`` endpoint), so
    the issue never got dispatched.

    We invoke with ``InvocationType="Event"`` instead: Lambda enqueues the
    event and returns immediately with HTTP 202 and an EMPTY payload, without
    waiting for the function to run. The kickoff step therefore finishes in
    seconds regardless of how long the agent takes.

    Because the invoke is async there is no function result to inspect:
    ``FunctionError`` and the response ``Payload`` only apply to a synchronous
    ``RequestResponse`` invoke, so we do not read or require them here. We still
    return the ``{"status_code", "response"}`` shape ``start_agent`` consumes;
    ``response`` is an empty dict since the run has not produced anything yet.
    """
    client = boto3.client("lambda", region_name=region)
    resp = client.invoke(
        FunctionName=function_name,
        InvocationType="Event",
        Payload=json.dumps(payload).encode("utf-8"),
    )
    # Async invoke: success is HTTP 202 (Accepted). Anything else is a
    # dispatch-time failure (throttling, auth, bad function name, ...); Lambda
    # may include an error description in the response Payload for those.
    status = resp.get("StatusCode")
    if status != 202:
        raw = ""
        body = resp.get("Payload")
        if body is not None and hasattr(body, "read"):
            raw = body.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"dispatcher Lambda async invoke failed with StatusCode={status}: {raw}"
        )
    return {"status_code": status, "response": {}}


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


def resolve_base_branch(issue: dict, override: str, default_base: str) -> str:
    """Resolve the base branch: dispatch override > base:<label> > default."""
    if override:
        return override
    for label in issue.get("labels") or []:
        name = label.get("name") if isinstance(label, dict) else label
        if name and name.lower().startswith(BASE_LABEL_PREFIX):
            candidate = name[len(BASE_LABEL_PREFIX):].strip()
            if candidate:
                return candidate
    return default_base


def start_agent(
    issue: dict,
    repo: str,
    comment_token: str,
    function_name: str,
    region: str,
    base_override: str,
    default_base: str,
) -> None:
    number = issue["number"]
    title = issue.get("title") or ""
    if not title.lower().startswith(TITLE_PREFIX):
        print(f"skip #{number}: title does not start with {TITLE_PREFIX!r}")
        return
    if issue.get("pull_request"):
        print(f"skip #{number}: pull request")
        return
    if issue.get("state") != "open":
        print(f"skip #{number}: issue is {issue.get('state')}")
        return
    if already_started(repo, number, comment_token):
        print(f"skip #{number}: already kicked off")
        return

    body = issue.get("body") or "(no description)"
    base_branch = resolve_base_branch(issue, base_override, default_base)

    payload = {
        "mode": "issue",
        "issue_number": number,
        "title": title,
        "body": body,
        "base_branch": base_branch,
        "repo": repo,
    }
    # Claim the issue BEFORE dispatching: the marker is the dedup key the 5-minute
    # poll checks, so posting it first means a failed comment can never cause a
    # re-dispatch loop. If the dispatch itself fails, the claim is withdrawn so
    # the next poll retries.
    claim = github_request(
        f"{GITHUB_API}/repos/{repo}/issues/{number}/comments",
        comment_token,
        {"body": (
            f"{KICKOFF_MARKER}\n"
            f"Dispatching issue #{number} to the AgentCore coding agent "
            f"(Lambda `{function_name}`, region `{region}`, base `{base_branch}`). "
            f"It will plan the work, implement and test each sub-task, and open a PR "
            f"against `{base_branch}` — or comment here if it cannot."
        )},
    )
    print(
        f"invoking dispatcher Lambda {function_name!r} in {region} for #{number} "
        f"(base_branch={base_branch})"
    )
    try:
        result = invoke_dispatcher(function_name, region, payload)
    except Exception:
        if isinstance(claim, dict) and claim.get("id"):
            github_request(
                f"{GITHUB_API}/repos/{repo}/issues/comments/{claim['id']}",
                comment_token,
                method="DELETE",
            )
        raise
    print(f"dispatched #{number} -> status {result.get('status_code')}")


def scan_project(
    owner: str,
    number: int,
    repo: str,
    project_token: str,
    comment_token: str,
    function_name: str,
    region: str,
    default_base: str,
    owner_type: str,
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
            start_agent(issue, repo, comment_token, function_name, region, "", default_base)
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
    function_name = env("DISPATCHER_LAMBDA_NAME", "agentcore-decomposer-dispatcher")
    region = env("AWS_REGION", "us-east-1")
    default_base = env("DEFAULT_BASE_BRANCH", "main")
    base_override = env("BASE_BRANCH_INPUT")
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
    if not function_name:
        print("DISPATCHER_LAMBDA_NAME is required", file=sys.stderr)
        return 1

    if event == "issues":
        label = env("LABEL_NAME")
        issue_number = env("ISSUE_NUMBER")
        if normalize(label) not in {"inprogress", "agent", "agentrun"}:
            print(f"ignore label {label!r}")
            return 0
        issue = fetch_issue(repo, int(issue_number), comment_token)
        start_agent(issue, repo, comment_token, function_name, region, base_override, default_base)
        return 0

    if event == "workflow_dispatch" and env("ISSUE_NUMBER"):
        issue = fetch_issue(repo, int(env("ISSUE_NUMBER")), comment_token)
        start_agent(issue, repo, comment_token, function_name, region, base_override, default_base)
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
        function_name,
        region,
        default_base,
        owner_type,
    )
    print(f"dispatched {started} agent(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

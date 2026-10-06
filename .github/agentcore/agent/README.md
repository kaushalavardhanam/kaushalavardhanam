# AgentCore coding agent (runtime container)

The container the **Bedrock AgentCore Runtime** runs for `agent-*` issues and
`/agent fix` PR comments. A Python orchestrator drives the **Claude Agent SDK**
(Claude Code, on Amazon Bedrock) through guarded sessions. The end-to-end
diagrams are in the root [README](../../../README.md#cloud-coding-agents-on-the-project-board).

## Files

| File | Purpose |
|---|---|
| `agent.py` | AgentCore entrypoint. Validates the payload, de-dups per job, registers an async task (`/ping` → `HealthyBusy`), runs the job on a background thread, returns `accepted` at once, and calls `StopRuntimeSession` when the job ends. |
| `jobs.py` | The two jobs: **issue → PR** and **`/agent fix` → commits on the PR**. Re-checks the launch conditions against GitHub, clones, runs the orchestrator, pushes, opens/replies with the report. The only place the GitHub token is used. |
| `orchestrator.py` | Requirements → baseline → plan (validated, re-planned on violations) → per-sub-task implement / revert out-of-scope / acceptance / commit, re-splitting failures → verify. |
| `plan.py` | JSON schemas for the requirements and plan steps, and the deterministic `validate_plan` rules. |
| `claude_runner.py` | One fresh Claude Code session per step (`claude_agent_sdk.query`), with the `PreToolUse` guard (`check_tool` / `check_bash`), budgets, and Bedrock env. |
| `git_pr.py` | Clone (token stripped from the checkout), push (token for that one command), GitHub REST helpers. |
| `github_app.py` | GitHub App secret → RS256 JWT → 1-hour installation token. |
| `requirements.txt` | Agent deps: `claude-agent-sdk` (bundles the Claude Code CLI), `bedrock-agentcore`, boto3, PyJWT, requests. |
| `requirements-test-env.txt` | Target-repo test deps, baked into a separate venv `/opt/testenv` used by the agent's Bash tool and by acceptance/verify commands. |

## How the decomposition is kept honest

The model plans; **code decides whether the plan is acceptable** (`plan.validate_plan`):

- the requirements step extracts numbered requirements from the ask (one per
  listed case / reported bug / "Done when" item), and **every requirement must
  be covered** by a sub-task;
- **every sub-task changes 1–4 files** — no "inspect" or "run the tests" steps
  (exploring happens while planning; verifying is each sub-task's acceptance
  command);
- every sub-task has **one runnable acceptance command** (pytest / unittest,
  with a cwd; `py_compile` is allowed only for the requirements' verify
  commands) that the orchestrator runs itself — the model's own claim of
  success is never trusted;
- acceptance is **fail-before / pass-after**: once it passes, the orchestrator
  stashes the sub-task's non-test changes and re-runs it; if it still passes,
  the sub-task is rejected ("test does not exercise the change") and retried,
  then re-planned. Test-only sub-tasks skip this check;
- dependencies point only backwards, and two sub-tasks touching the same file
  must be ordered.

A rejected plan goes back to the planner with the violation list (3 attempts).
A sub-task that fails its acceptance check twice is re-planned into smaller
sub-tasks (depth ≤ 2). Whatever still fails is reported, and the PR is opened as
a **draft** with the passing commits.

## Guardrails

| Risk | Control |
|---|---|
| Edits outside the planned scope | `PreToolUse` denies Edit/Write outside the sub-task's files; the orchestrator also reverts any other change after each session and stages only declared paths. |
| Shell misuse | Bash allowlist (read tools, `python`/`pytest`, read-only `git`); no command substitution, redirection, `git commit/push`, network tools, or `pip install`. |
| GitHub credential exposure | The token never enters a Claude session's env or the checkout (`origin` is rewritten token-free after clone). |
| Runaway spend | Per-session `max_budget_usd` ($3) and `max_turns`; per-job `AGENT_BUDGET_USD` (Terraform `agent_budget_usd`, default $20). |
| Stray / duplicate launches | Dispatcher: `agent-*` title gate + one runtime session id per issue / fix comment; runtime: re-checks issue/PR state, in-session de-dup, `Agent-Fix-Comment` trailer idempotency. |

**Residual risk.** The hook is a policy, not a sandbox: a test run executes repo
code, which runs with the runtime's execution role (Bedrock invoke, the GitHub
App secret). Inputs are limited to trusted authors (board access / write
collaborators / the issue author), and every session is an isolated AgentCore
microVM that is stopped when the job ends.

## Runtime contract

Environment (set by `../terraform/main.tf` and the Dockerfile):

| Env var | Meaning |
|---|---|
| `BEDROCK_MODEL_ID` | Anthropic model / inference profile Claude Code runs on (default `global.anthropic.claude-sonnet-5-5`; also used as the fast model). |
| `GITHUB_APP_SECRET_ARN` | Secrets Manager secret with the GitHub App credentials. |
| `GITHUB_REPO` | Default `owner/name`. |
| `AGENT_BUDGET_USD` | Per-job model spend cap. |
| `TEST_VENV_BIN` | `/opt/testenv/bin`, prepended to `PATH` for Claude's Bash and the acceptance commands. |

The GitHub App secret is a JSON string `{"app_id", "installation_id", "private_key"}`
(PEM, PEM with `\n` escapes, or base64 PEM).

## Build and deploy

AgentCore Runtime requires **linux/arm64** images. `../deploy.sh <account-id>`
builds the image in CodeBuild and applies the Terraform.

## Local checks

```bash
python -m venv /tmp/agentcore-venv && /tmp/agentcore-venv/bin/pip install -r requirements.txt pytest
/tmp/agentcore-venv/bin/python -m pytest -q          # agent tests (offline)
```

The tests replace Claude with a scripted runner, but run git, the plan
validator, acceptance commands, commits and pushes (to a local bare repo) for
real.

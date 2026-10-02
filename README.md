# kaushalavardhanam
This is a group created with an intent to up-skill members while enabling them to build something valuable and long lasting. The group includes Engineering students, IT Professionals, SamskritaBharati Volunteers and aspirants who intend to learn and wear multiple hats.

This repository is for bootstrapping initiatives being led under Kaushalavardhanam. Students/ aspirants can use this as inspiration to continue building in the project specific repositories

## Ideation Document
The purpose of the projects for various cohorts in this repo are meant to provide seed resources to ensure projects proposed are feasible, achievable within stipulated period that students are able to commit to and to ensure the cohorts are successful in completing the products without having to evaluate multiple paths. 

- Cohort 1 - ZatamOnAWS - This document has the actual game build developed in ZatamOnAWS (Temporary name) folder. Final project is available https://github.com/skopp002/Sanskrit_Family_Feud_GameShow/tree/Survey
- Cohort 2 - [speaking_buddy]([speaking_buddy](https://github.com/skopp002/kaushalavardhanam/blob/main/speaking_buddy/README.md)/) - A Streamlit-based Luxembourgish pronunciation learning tool with Praat-based phonetic analysis
- Cohort 3 - [mitra](mitra/) - A multilingual conversational robot with vision and audio capabilities for Sanskrit and Kannada

## Cloud coding agents on the project board
Issues titled `agent-*` on the [kaushalavardhanam org project board](https://github.com/orgs/kaushalavardhanam/projects/1) get picked up automatically when moved to **In Progress**: an agent plans the issue, implements and tests it, and opens a PR. Reviewers can then comment `/agent fix <what failed>` on that PR and the agent pushes a fix to the same branch.

| Backend | Status | Where the agent runs | Workflow |
|---|---|---|---|
| **AgentCore + Claude Code** | **active** | A Bedrock AgentCore Runtime session (serverless microVM) running the Claude Agent SDK on Amazon Bedrock | `agentcore-agent-in-progress.yml`, `agentcore-agent-pr-feedback.yml` |
| Cursor Cloud Agent on an AgentCore worker pool | dormant (manual only) | A long-lived self-hosted Cursor worker hosted on AgentCore | `cursor-agent-in-progress.yml` |
| Claude Code on GitHub Actions | removed | A GitHub-hosted runner (`anthropics/claude-code-action`) | — |

### 1. How the kickoff watches the board

```mermaid
flowchart TD
    cron["schedule: every 5 min"] --> scan["GraphQL: list project items"]
    label["issues: labeled<br/>in-progress / agent / agent-run<br/>(AgentCore only)"] --> fetch
    manual["workflow_dispatch<br/>issue_number"] --> fetch
    scan --> inprog{"Status = In Progress?"}
    inprog -- no --> stop1(["ignore"])
    inprog -- yes --> fetch["fetch the issue"]
    fetch --> gate{"title starts with agent-<br/>open, not a PR<br/>no kickoff comment yet?"}
    gate -- no --> stop2(["skip"])
    gate -- yes --> claim["post kickoff comment<br/>(the dedup marker)"]
    claim --> dispatch["start the backend's worker"]
    dispatch -- "dispatch failed" --> unclaim["delete the kickoff comment<br/>so the next poll retries"]
```

Both kickoffs apply the same `agent-*` / In Progress filter; while Cursor is manual-only, only the AgentCore kickoff polls the board and answers labels. The AgentCore kickoff posts its kickoff comment *before* dispatching, so a later poll never starts the same issue twice, and withdraws it only if the dispatch itself fails. (The dormant Cursor kickoff still comments after starting the agent and does not check that the issue is open.) The base branch is `main`, a `base:<branch>` label on the issue, or the `base_branch` input of a manual run.

### 2. AgentCore + Claude Code (active)

```mermaid
sequenceDiagram
    autonumber
    participant GA as GitHub Actions<br/>(kickoff)
    participant L as Dispatcher Lambda
    participant R as AgentCore Runtime session
    participant C as Claude Code sessions<br/>(Agent SDK on Bedrock)
    participant GH as GitHub

    GA->>GA: OIDC -> assume agentcore-decomposer-github-oidc
    GA->>L: async invoke {mode: issue, issue, base_branch}
    L->>R: invoke_agent_runtime(sessionId = one per issue)
    R-->>L: accepted (work continues in background, ping = HealthyBusy)
    R->>GH: re-check: issue open, title agent-*, branch not taken
    R->>GH: clone base branch (token stripped from the checkout)
    R->>C: REQUIREMENTS (read-only) -> R1..Rn, verify commands, constraints
    R->>R: run verify commands = baseline
    loop until the plan passes validation (max 3)
        R->>C: PLAN (read-only, explores the repo) -> sub-tasks
        R->>R: validate: every Rn covered, each sub-task changes 1-4 files,<br/>runnable acceptance command, dependencies point backwards
    end
    loop each sub-task, in order
        R->>C: IMPLEMENT (may edit only that sub-task's files)
        R->>R: revert out-of-scope edits, run acceptance command
        alt passes
            R->>R: commit
        else fails twice
            R->>C: re-plan just this sub-task into smaller ones (max depth 2)
        end
    end
    R->>R: re-run verify commands, compare with baseline
    R->>GH: push agentcore/<slug>-<n>, open PR (draft if anything failed)<br/>with a requirements / sub-task / verification report
    R->>R: StopRuntimeSession (no idle compute left behind)
```

Nothing runs between jobs: Actions minutes are spent only on the 5-minute board poll, and a runtime microVM exists only while a job is in flight. The Claude sessions never hold the GitHub token, cannot `git commit`/`push`, use the network tools, or edit files outside their sub-task.

### 3. Reviewer feedback: `/agent fix`

```mermaid
sequenceDiagram
    autonumber
    actor Rev as Reviewer
    participant GA as GitHub Actions<br/>(agentcore-agent-pr-feedback)
    participant L as Dispatcher Lambda
    participant R as AgentCore Runtime session
    participant GH as GitHub

    Rev->>GH: comment "/agent fix<br/>pytest tests/test_vocabulary.py fails at import: ..."
    GH->>GA: issue_comment (runs from main)
    GA->>GA: allow only the linked issue's author or a write collaborator,<br/>only on an open in-repo agentcore/* PR
    GA->>GH: post "On it" (dedup marker) + 👀
    GA->>L: async invoke {mode: fix, pr_number, comment_id}
    L->>R: invoke_agent_runtime(sessionId = one per comment)
    R->>GH: re-check the PR, read the comment, inline review comments,<br/>linked issue, changed files
    R->>R: clone the PR branch, skip if "Agent-Fix-Comment: <id>" already landed
    R->>R: same requirements -> plan -> implement -> verify loop,<br/>with the reviewer's failure as the ask
    R->>GH: push commits to the SAME branch, reply with the report
    R->>R: StopRuntimeSession
```

### 4. Cursor Cloud Agent on an AgentCore worker pool (dormant)

```mermaid
sequenceDiagram
    autonumber
    participant GA as GitHub Actions<br/>(cursor-agent-in-progress)
    participant CA as Cursor API<br/>api.cursor.com/v1/agents
    participant W as Self-hosted Cursor worker<br/>(AgentCore Runtime session)
    participant GH as GitHub

    Note over W: must already be running and HealthyBusy<br/>(started separately, see .github/WORKER_GIT.md)
    GA->>CA: create agent {prompt, env: pool agentcore-platform-agents,<br/>repo, startingRef, autoCreatePR}
    CA-->>GA: 403 feature_unavailable today:<br/>private workers are not enabled on this Cursor plan
    CA->>W: assign the agent run to the pool
    W->>GH: clone, implement, run tests
    W->>GH: push with CURSOR_GIT_TOKEN, Cursor opens the PR
```

It is manual-only (`workflow_dispatch` with an issue number): the `schedule` trigger is commented out, it no longer answers labels, the `CURSOR_API_KEY` secret has expired, and Cursor has not enabled private workers on this plan. Unlike the AgentCore backend, the worker is a long-lived session that has to exist *before* work arrives, and it needs its own GitHub PAT on the worker. The worker image and Terraform live in [cursor-cookbook](https://github.com/skopp002/cursor-cookbook/tree/main/self-hosted-cloud-agent/agentcore).

### 5. Claude Code on GitHub Actions (removed)

```mermaid
sequenceDiagram
    participant GA as GitHub Actions<br/>(claude-agent-in-progress)
    participant GW as GitHub Actions<br/>(claude-agent-issue)
    participant GH as GitHub
    GA->>GW: workflow_dispatch {issue_number, base_branch}
    GW->>GW: OIDC -> github-actions-claude-bedrock role
    GW->>GW: anthropics/claude-code-action on the runner (one long prompt)
    GW->>GH: Claude creates the branch + PR with GITHUB_TOKEN
```

Removed because the AgentCore backend now runs the same Claude Code agent and adds what this one lacked: a validated plan with requirement coverage, per-sub-task acceptance checks, draft PRs instead of silent partial work, and the `/agent fix` loop. It also answered the same labels as the AgentCore backend, so labelling one issue started two agents racing to open PRs. PRs it opened with `GITHUB_TOKEN` also never triggered CI.

### 6. Swapping Claude Code for Kiro inside AgentCore (not implemented)

What would change to run Kiro instead of Claude Code in backend 2. Only the coding engine changes. The board watcher, `/agent fix`, dispatcher, session-per-job lifecycle, orchestrator, plan validator, git/PR code and Terraform stay as they are, because `orchestrator.py` only calls `runner.run(prompt, write_paths=..., schema=...)`. Checked against `kiro-cli` 2.27.0 and `kirocrew` as installed here; re-check the flags against the version you deploy.

**Pick the engine: `kiro-cli`, not Kiro Crew.**
- **`kiro-cli chat --no-interactive`** runs one headless session and exits, which matches one fresh session per step. Use this.
- **Kiro Crew (`kirocrew`)** is a personal agent *gateway*: a long-running server with a dashboard, memory and messaging channels. Its `kirocrew cloud` mode provisions your own EC2 instance, which is always-on rather than serverless. `kirocrew run TASK.md` runs a whole task with its own planner and test loop. Using it would replace `orchestrator.py`, along with the requirement-coverage and acceptance checks that `plan.py` enforces in code.

**What to update**

| Piece | With Claude Code (today) | With Kiro |
|---|---|---|
| Session runner | `claude_runner.py` → `claude_agent_sdk.query()` | New `kiro_runner.py` with the same `run()` signature. It runs `kiro-cli chat --no-interactive --output-format stream-json --agent <generated> "<prompt>"` with `cwd` = the checkout, and reads the JSON Lines events for the final answer. |
| Authentication | `CLAUDE_CODE_USE_BEDROCK=1` + the runtime's IAM execution role | **Blocker.** `kiro-cli login` only signs in a *person* (Builder ID / social, or IAM Identity Center for Pro) through a browser or OAuth device flow. There is no IAM-role or workload-identity login, so a fresh microVM per job cannot authenticate unattended. This is the gap filed in `mitra/kiro-cloud-agents-pfr.md`. Until Kiro supports it, the only workaround is copying a signed-in user's token into Secrets Manager and refreshing it, which runs every job as that person. |
| Model + IAM | `BEDROCK_MODEL_ID`; `bedrock:InvokeModel` on Claude Sonnet 5.5 | Inference goes through Kiro's service under the signed-in identity, so `local.bedrock_model_ids` / `agentcore_runtime_bedrock` in `main.tf` would no longer be needed. Pick the model with `--model` (`kiro-cli chat --list-models` lists what the account can use). |
| Structured output (requirements, plan) | `output_format` JSON schema → `ResultMessage.structured_output` | No schema-validated output option. Ask for a single JSON object in the final message, validate it with `jsonschema` against `plan.REQUIREMENTS_SCHEMA` / `PLAN_SCHEMA`, and treat a parse failure as a plan violation. The planner already retries up to 3 times with the violation list. |
| Tool guard | Python `PreToolUse` hook calling `check_tool()` | Write a throwaway agent JSON per session. Set `tools` / `allowedTools` to `fs_read`, `fs_write`, `shell` (no `web_fetch`, no `subagent`) and `toolsSettings.shell.allowedCommands` to the Bash allowlist in `claude_runner.py`. Restrict write paths with an `fs_write` path setting or a `preToolUse` hook script that calls `check_tool()`. Agent files here only show `agentSpawn` / `userPromptSubmit` / `postToolUse` hooks, and `postToolUse` cannot block, so confirm blocking `preToolUse` support first. Pass `--trust-tools=<that list>`, never `--trust-all-tools`. The orchestrator's revert-out-of-scope step stays as the backstop either way. |
| Budgets | `max_turns`, `max_budget_usd`, `ResultMessage.total_cost_usd` | No per-session dollar cap. Enforce a wall-clock limit (`subprocess` timeout) per session and count events from the stream; `AGENT_BUDGET_USD` would become a time or step budget. |
| Repo settings | `setting_sources=["project"]` (repo `CLAUDE.md`) | Kiro reads project steering/specs from `.kiro/`; commit those per project instead of a `CLAUDE.md`. |
| Image | `claude-agent-sdk` wheel (bundles the Claude Code CLI) | Install the linux/arm64 `kiro-cli` binary in the `Dockerfile` instead and drop `claude-agent-sdk` from `requirements.txt`. The rest of the image (test venv, git, ripgrep) is unchanged. |
| Tests | `ScriptedRunner` stands in for Claude | Unchanged: the orchestrator and job tests never call the engine. Add `kiro_runner` tests for building the CLI arguments and parsing the event stream. |

**Bottom line:** the code changes are mostly confined to one new runner module plus the Dockerfile. What blocks it is unattended authentication, not the code. Until `kiro-cli` can sign in with an IAM role, Kiro can't run in a serverless, session-per-job runtime without borrowing a person's login.

**Conventions:**
- Title the issue `agent-<short description>` for it to be picked up.
- By default the agent branches off and opens its PR against `main`. To target a different branch, add a `base:<branch-name>` label to the issue (e.g. `base:release-1.2`), or pass `base_branch` explicitly on a manual `workflow_dispatch` run.
- Write a "Done when" section with runnable commands (e.g. `cd mitra && pytest tests/test_x.py -q`): the agent turns it into its verify step.
- On the PR, `/agent fix` followed by what failed (the command and the error) gets a fix pushed to the same branch.

**Security note:** this repo is public, so anyone can open an issue — including one titled `agent-*` with a crafted body. That alone doesn't trigger anything; it still takes someone with repo write/triage access adding the label, or someone with project-board access moving it to In Progress, and `/agent fix` is honoured only from the linked issue's author or a write collaborator. Because the issue body and fix comments become instructions to an agent that can push branches and spend Bedrock budget, **read the issue body before labeling it or moving it to In Progress** — don't triage on title alone.

## code_with_q_cli
This has a langgraph based multiagent orchestration application to enable code generation based on prompts

## Resources to build the games
https://aws.amazon.com/blogs/gametech/online-multiplayer-amazon-gamelift-aws-serverless/

For Python Virtual Env:
1.  cd code_with_q_cli
2.  Explore the modules and utilize available resources to continue the game development based on the finalized design


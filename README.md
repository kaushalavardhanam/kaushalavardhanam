# kaushalavardhanam
This is a group created with an intent to up-skill members while enabling them to build something valuable and long lasting. The group includes Engineering students, IT Professionals, SamskritaBharati Volunteers and aspirants who intend to learn and wear multiple hats.

This repository is for bootstrapping initiatives being led under Kaushalavardhanam. Students/ aspirants can use this as inspiration to continue building in the project specific repositories

## Ideation Document
The purpose of the projects for various cohorts in this repo are meant to provide seed resources to ensure projects proposed are feasible, achievable within stipulated period that students are able to commit to and to ensure the cohorts are successful in completing the products without having to evaluate multiple paths. 

- Cohort 1 - ZatamOnAWS - This document has the actual game build developed in ZatamOnAWS (Temporary name) folder. Final project is available https://github.com/skopp002/Sanskrit_Family_Feud_GameShow/tree/Survey
- Cohort 2 - [speaking_buddy]([speaking_buddy](https://github.com/skopp002/kaushalavardhanam/blob/main/speaking_buddy/README.md)/) - A Streamlit-based Luxembourgish pronunciation learning tool with Praat-based phonetic analysis
- Cohort 3 - [mitra](mitra/) - A multilingual conversational robot with vision and audio capabilities for Sanskrit and Kannada

## Cloud coding agents on the project board
Issues titled `agent-*` on the [kaushalavardhanam org project board](https://github.com/orgs/kaushalavardhanam/projects/1) get picked up automatically when moved to **In Progress**: an agent implements the issue and opens a PR.

Two agent systems are wired up:
- **Claude Code agent** (active) — runs via `anthropics/claude-code-action`, billed through Amazon Bedrock using global CRIS (Cross-Region Inference), authenticated with a GitHub OIDC-federated IAM role (no static AWS/Anthropic secret in this repo). Workflows: `.github/workflows/claude-agent-in-progress.yml` (polls the board every 5 minutes, also runs on `issues: labeled`) dispatches `.github/workflows/claude-agent-issue.yml` (implements the issue via `.github/scripts/kick_claude_agent.py`).
- **Cursor Cloud Agent** (dormant) — same trigger convention, but its scheduled polling is disabled because the configured self-hosted worker pool isn't enabled on the current Cursor plan. Workflow: `.github/workflows/cursor-agent-in-progress.yml` via `.github/scripts/kick_cursor_agent.py`. Retrigger manually via `workflow_dispatch` (with an issue number) once a working key/pool is in place, or re-enable the `schedule` trigger in that file.

**Conventions:**
- Title the issue `agent-<short description>` for it to be picked up.
- By default the agent branches off and opens its PR against `main`. To target a different branch, add a `base:<branch-name>` label to the issue (e.g. `base:release-1.2`), or pass `base_branch` explicitly on a manual `workflow_dispatch` run.

## code_with_q_cli
This has a langgraph based multiagent orchestration application to enable code generation based on prompts

## Resources to build the games
https://aws.amazon.com/blogs/gametech/online-multiplayer-amazon-gamelift-aws-serverless/

For Python Virtual Env:
1.  cd code_with_q_cli
2.  Explore the modules and utilize available resources to continue the game development based on the finalized design


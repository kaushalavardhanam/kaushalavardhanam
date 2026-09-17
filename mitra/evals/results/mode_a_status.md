# Mode A run inventory (issue #9)

Issue #7 asked for three spoken runs of each of the ten scenarios, offline and Bedrock.

| Run | Provider | Model | Region | Orchestrator | Status |
|---|---|---|---|---|---|
| 1 | ollama | qwen3-vl:8b-instruct | null | custom | **Recorded** — 2026-09-16 operator log. Gate fail (2.2 / 2.2). See `mode_a_run1_2026-09-16.md`. |
| 2 | ollama | qwen3-vl:8b-instruct | — | custom | **Not run** — no MuJoCo daemon, no mic, 3.7 GiB RAM. |
| 3 | ollama | qwen3-vl:8b-instruct | — | custom | **Not run** — same. |
| 1–3 | bedrock | e.g. us.anthropic.claude-sonnet-4-6 | us-west-2 | custom | **Not run spoken.** Written Mode B with this id exists (`region=us-west-2`, cost recorded, `ollama_loaded=false`). |
| 1–3 | either | either | — | pipecat | **Not run spoken.** Inject + echo tests only. |

Repeating the 2026-09-16 transcripts through the orchestrator three times would not be a new spoken run and is not counted here.

`EVALUATION.md` and ADR-001 no longer say “Mode A pending a Mac” as if run 1 did not exist. They say run 1 exists and failed, and runs 2–3 plus Bedrock spoken are still Mac-only.

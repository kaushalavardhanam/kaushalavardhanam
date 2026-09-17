# ADR-001 — Default mode, orchestrator, and model IDs

**Status:** Accepted for issue #9 (closes the #7 deferral)  
**Date:** 2026-09-17  
**Config default in `config.yaml`:** custom orchestrator + local Ollama (offline / privacy)  
**Recommended mode for the Sanskrit quality gate:** Bedrock + Claude Sonnet 4.6, prompt v1  
**Opt-in:** `--orchestrator pipecat`, `--llm-provider bedrock`

## Context

PR #8 delivered Bedrock, a Pipecat PoC, diagnostics, and a written bake-off, then deferred the default-mode decision pending Mode A and vision. Issue #9 recorded the first live Mode A run (2026-09-16, offline Qwen) and required that decision to be written down.

## Decision 1 — Orchestrator: keep custom

**Adopt Pipecat as the default? No.** Unchanged from #8.

Evidence: unit + inject parity, echo-gate tests on both paths, no spoken Pipecat-vs-custom A/B. Daily transports would leave the privacy boundary. See `evals/results/pipecat_e2e.md`.

## Decision 2 — Cloud vs local (the #7 deferral)

**Recorded, not deferred.**

| Question | Decision | Evidence |
|---|---|---|
| Does the offline default meet the #7 Sanskrit gate? | **No.** | Mode A run 1 (2026-09-16): grammar 2.2 / semantic 2.2, Hindi on three turns, greeting closer on every turn. Eight of ten scores below 3. |
| What is offline mode for? | Privacy, air-gapped demo, no AWS. It remains installed and is still `config.yaml` default so a laptop without credentials works. | Local-first is a load-bearing product rule (DESIGN / CLAUDE.md). |
| What should an operator use when the quality gate matters? | **Bedrock**, model `us.anthropic.claude-sonnet-4-6` (or `us.openai.gpt-5.6-sol`), **prompt v1**. | Mode B written gate passed (Sonnet 4.6: 4.5/4.8; Sol: 4.6/4.8). Nova Pro 3.4 grammar **fail**. Haiku 4.5 3.6 + non-Devanagari **fail**. |
| Was that gate measured on spoken Mode A? | **No.** Spoken Bedrock Mode A was not run (no daemon/mic). The recommendation is Mode B + the fact that offline Mode A failed. | `mode_a_status.md` |

`config.yaml` is **not** flipped to Bedrock in this PR. Flipping it would break offline-first without credentials. The ADR is the recommendation; the CLI is `--llm-provider bedrock --llm-id us.anthropic.claude-sonnet-4-6`.

Prompt v2 (Hindi bans, no greeting closer) is in the tree to attack the Qwen failure modes. It was **not** evaluated on Qwen (no Ollama). On Sonnet it *regressed* the sports turn (English self-correction + `खेलः`). Do not report v2 as the configuration that cleared the gate.

## Decision 3 — Selected IDs

Exact IDs. None are hard-coded as the only legal value in config.

| Role | Selected ID | Why | Evidence type |
|---|---|---|---|
| Conversation (quality gate) | `us.anthropic.claude-sonnet-4-6` | Written gate pass; faster/cheaper than Sol on this worker | Mode B 2026-09-08 |
| Conversation (alternate) | `us.openai.gpt-5.6-sol` | Written gate pass; rejects `temperature`; high tail latency | Mode B 2026-09-08 |
| Conversation (offline) | `qwen3-vl:8b-instruct` | Only local VLM with tools in-tree | Mode A run 1 **fails** the gate |
| Vision (quality) | `us.anthropic.claude-sonnet-4-6` | Same id; more grounded than Nova on identical stand-in JPEGs | Synthetic vision 2026-09-17 |
| Vision (offline) | `qwen3-vl:8b-instruct` | Unscored here | — |
| ASR | `mlx-community/whisper-large-v3-turbo` on Mac; whisper-small/transformers was what 2026-09-16 actually ran | English WER 0.048 on the ten items | Mode A run 1 + config |
| VAD / wake | Silero (fixed `min_speech_s`) / transcript “mitra” on whisper-small | Code + #8 diagnosis | Tests |
| TTS | Indic Parler-TTS primary; `facebook/mms-tts-hin` is what 2026-09-16 actually used | Hindi VITS is a known limitation | Mode A run 1 |

## Decision 4 — Recognition

#7’s “simulator recognizes sentences incorrectly” objective is **restated, not treated as the blocking LLM bug**. Nine of ten English transcripts were exact; micro-WER **0.048**. Residual: `Do you play sports?` → `2 play sports` (likely VAD truncation / Do→2). `first_error_stage` is now `asr` for a leading-digit hypothesis; English retry treats it as unusable.

## Assumptions vs measurements

**Measured**

- Offline Mode A run 1 quality (2.2 / 2.2) and WER (0.048)
- Mode B written scores for Nova, Sonnet 4.6, Haiku 4.5, GPT-5.6 Sol (prompt v1)
- Sonnet 4.6 prompt-v2 sports leak
- Quality-stage precision 0.875 / recall 1.0 on the ten scored turns (target 1,2,3,4,5,8,9)
- Bedrock `region=us-west-2`, token cost, `ollama_loaded=false`
- Inject-path RSS +0.6 MB over 30 turns
- Echo-gate unit tests (custom + Pipecat)

**Assumptions / not measured**

- Spoken Bedrock Mode A would match written Sonnet quality
- Prompt v2 would lift Qwen (not run)
- MuJoCo camera JPEGs would match the synthetic stand-ins
- Parler-TTS is more Sanskrit-appropriate than VITS Hindi (not A/B’d)
- MLX large-v3-turbo is faster than transformers whisper-small at equal WER (not A/B’d on this host)
- 950 MB / 10-turn live growth is cache, not a leak (inject path does not reproduce it)

## Human linguistic review

Register, food-persona honesty, and any score below 5 still need a Sanskrit reviewer. Agreement and Hindi items in the 2026-09-16 table are high confidence.

## Consequences

- Operators who need the #7 gate use Bedrock Sonnet 4.6 (prompt v1) and accept cloud cost + a non-null Region.
- Offline remains; it will still fail the gate with current Qwen until a local model is re-baked.
- `validation_ok` is script-only. Results must report `quality_ok` separately.
- Pipecat stays a flag.

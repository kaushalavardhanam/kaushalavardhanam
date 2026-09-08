# ADR-001 — Pipecat PoC and Bedrock inference mode

**Status:** Accepted for the evaluation branch (issue #7)  
**Date:** 2026-09-08  
**Default in `config.yaml`:** custom orchestrator + local Ollama  
**Opt-in:** `--orchestrator pipecat`, `--llm-provider bedrock`  
**Branch:** `cursor/agent-mitra-pipecat-cloudllm-eecf`

## Context

Mitra’s custom orchestrator (`src/orchestrator.py`) already implements wake, barge-in, VAD, ASR, Strands tools, Devanagari validation, lexicon override, TTS, and `flush_mic` after playback. Issue #7 asked whether Pipecat should replace that layer and whether Amazon Bedrock should replace local Qwen3-VL for conversation and vision.

## Decision 1 — Orchestrator: keep custom; retain Pipecat as a flag

**Adopt Pipecat as the default? No.**

Evidence:

1. The PoC (`src/orchestration/pipecat_runtime.py`) maps audio → processors → the *same* `handle_event` path. Domain logic is not rewritten (issue constraint).
2. Daily Pipecat’s stock transports (WebRTC / Daily / cloud STT-TTS) would move raw audio off-host or replace Indic TTS. That violates the privacy model and Sanskrit TTS requirement.
3. Pipecat is async and service-oriented; Mitra is a single-threaded state machine with two daemon helpers. A full replacement would re-implement barge-in, silence timeout, and playback flush with a high regression risk.
4. Unit tests on `PipecatOrchestrator` show wake, validation, and barge-in parity with the custom engine when events are injected. That is necessary but not sufficient to retire the custom loop.
5. The live audio pump now **queues** wake/utterance events for the existing `handle_event` run loop (no `handle_event` from the mic thread). That matches the custom engine’s concurrency model.
6. End-to-end MuJoCo + spoken turns were **not** run in this Linux cloud worker (no Reachy daemon, no operator mic, 3.7 GiB RAM). The PoC is reversible and off by default.

**Keep:** `orchestration.engine: custom`.  
**Keep:** Pipecat path for further A/B on a Mac with the simulator.  
**Do not:** force a full replacement.

### What Pipecat replaces vs what remains

| Replaced | Remains |
|---|---|
| Mic pump if/else (`_audio_loop`) | Wake detectors |
|  | Silero / energy VAD |
|  | Local Whisper ASR |
|  | Language detector |
|  | Strands agent + tools |
|  | Validator + lexicon |
|  | Indic Parler-TTS / VITS |
|  | `flush_mic` / barge-in events |

## Decision 2 — LLM/VLM: Bedrock is a first-class explicit mode

**Default remains Ollama `qwen3-vl:8b-instruct` (offline).**  
**Bedrock mode** (`models.llm.provider: bedrock`) runs *all* conversational and vision inference through Bedrock and **does not** start, load, or contact Ollama/Qwen.

Rules:

- Standard AWS credential chain only. No keys in git.
- Region, model id, temperature, timeout, retries, max tokens are configurable.
- Failures raise `ProviderError` with an actionable hint. No silent model swap.
- `models.llm.fallback.enabled` is the only way to use another provider after a failure (default false).
- The older `cloud_fallback` block (validation-time) is unchanged and still default-off.

### Preferred Bedrock IDs (pending live Sanskrit gate)

| Role | ID | Why |
|---|---|---|
| Cost / latency shortlist | `us.amazon.nova-pro-v1:0` | VLM + Converse tools; CRIS from `us-west-2` |
| Quality shortlist | `us.anthropic.claude-sonnet-4-20250514-v1:0` | Stronger multilingual/tools expectation |
| Offline | `qwen3-vl:8b-instruct` | Unchanged local path |

Live invoke on this worker was **AccessDenied** (`bedrock:InvokeModel`). Do not treat research as a passed quality gate.

## Decision 3 — ASR / VAD stay local; recognition fixes

First-error analysis of the reported “simulator hears sentences wrong” problem (see EVALUATION.md):

- Not assumed to be the LLM.
- Confirmed code defects: diagnostic script used a different ASR/mic than `config.yaml`; Silero dropped `min_speech_s`; Whisper peak-normalized near-silence; known hallucinations accepted as transcripts.
- Fixes shipped; live WER still needs consented WAVs on the Mac.

## Consequences

- Operators choose Bedrock with CLI/config; laptops no longer need ~6 GB Qwen resident in that mode.
- Offline demo still works with `--llm-provider ollama`.
- Pipecat can be deleted later without touching domain modules.
- Sanskrit quality gate for a Bedrock default is **blocked** until Mode B runs with model access.

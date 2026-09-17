# Evaluation report — issue #7 close-out (issue #9)

**Baseline commit (origin/main):** `c4dea80f8334bbcb31560b52ea8eb2de3d437dcf`  
**Started from:** PR #8 head `9dbdf55` (not rebased)  
**This branch:** `cursor/agent-mitra-closeout-issue7-modea-sanskrit-gate-cac6`  
**Evaluator (Sanskrit rubric):** `cursor-grok-4.6-high-fast` (cloud agent). Not a candidate model.  
**PDF `tests/mitra-2026-08-22-1038-mobile.pdf`:** not present, not committed, not required. Corpus is `evals/corpus/conversation.yaml`.

Issue #9 is the unmet evidence from #7. PR #8 infrastructure is reused, not re-specified. `conversation.md` remains a **hand-written reference fixture** and is never counted as a model result.

## 1. Environment limits (measurements vs assumptions)

| Check | Result |
|---|---|
| Git baseline recorded | Yes — `evals/results/baseline.json` |
| Live Reachy Mini MuJoCo daemon | **No** — not installed / no display in this worker |
| Operator microphone | **No** |
| Local Ollama / `qwen3-vl:8b-instruct` | **No** |
| AWS identity | Assumed role `cursor_worker-execution-role`, `us-west-2` (account `146666888814`) |
| Worker RAM | **3.7 GiB** — local Qwen3-VL 8B cannot be loaded here |
| `bedrock:Converse` / `InvokeModel` | **OK** on `us.amazon.nova-pro-v1:0`, `us.amazon.nova-lite-v1:0`, `us.anthropic.claude-sonnet-4-6`, `us.anthropic.claude-haiku-4-5-20251001-v1:0`, `us.openai.gpt-5.6-sol` / `global.openai.gpt-5.6-sol`. Sol rejects `temperature` and requires `maxTokens` ≥ 16. Bare `openai.gpt-5.6-sol` needs an inference profile. Claude Sonnet 4 (`20250514`) is **legacy**. See `evals/results/bedrock_probe.json`. |
| `bedrock:ListFoundationModels` | Not required for invoke; last probe was AccessDenied (listing only). |
| EOL ids (Claude 3.5, Titan Text, Cohere Command Light, Llama 3.2 11B) | `ResourceNotFoundException` — retired |
| Unit tests | Run in this worker (`tests` minus `tests/hw`) |

Conclusions below separate **measured in this worker**, **measured in code/tests**, and **requires the operator Mac**.

## 2. Baseline (before provider/orchestrator swap)

Recorded from `config.yaml` at `c4dea80` / this branch’s config snapshot:

| Field | Value |
|---|---|
| Mic | `built_in` / `MacBook Pro Microphone` (Tahoe USB-mic workaround) |
| Sample rate target | 16 kHz; resample if robot rate differs |
| VAD | Silero, `min_silence_s` was 0.8, `min_speech_s` 0.3 **ignored by Silero factory** |
| ASR | `mlx-community/whisper-large-v3-turbo`, peak-normalize always |
| LLM | `ollama` / `qwen3-vl:8b-instruct` |
| TTS | Indic Parler-TTS, VITS Hindi fallback |
| Orchestrator | custom |
| Raw audio logged | No (FR-7.3) |

Live TTFA, e2e latency, and peak laptop RSS of a spoken turn **could not be sampled here**. The README’s prior M1 Max figure (~3 s warm Qwen, ~11.5 GB resident) remains the last operator-reported baseline.

Turn logs now include `provider`, `model_id`, `region`, `audio_stats` (rms/peak/duration only), `first_error_stage`, `validation_*`, `ttfa_s`, `e2e_s`, `rss_mb`.

## 3. Sentence recognition diagnosis

### Reproduction (code-level, not live speech)

The reported failure — simulator “recognizes some spoken sentences incorrectly” — was traced **before** blaming the LLM.

| Stage | Finding | Evidence |
|---|---|---|
| Capture | Tahoe robot USB mic can be all-zero (`reachy_mini#820`); app already uses `mic_source: built_in`. Diagnostic `scripts/test_audio.py` **ignored** that and opened the robot mic / default Transcriber. | Code read + bug: `wake` printed before assignment. |
| Capture | Peak-normalize of near-silence amplifies fan/echo into Whisper garbage. | `asr.py` before change; now `min_peak` skip. |
| VAD | `make_segmenter("silero")` **discarded `min_speech_s`**. Short noise or clipped words could become utterances. | `vad.py` `kwargs.pop("min_speech_s")`. |
| ASR | Whisper hallucinations (`Thanks for watching`, repetitions) accepted as transcripts → LLM answers the wrong question. | `hallucinations.py` + tests. |
| ASR | `condition_on_previous_text` default (engine) causes loops on short clips. | Now forced false. |
| Language | Devanagari→`sa`, Latin→`en` is fine for the ten English questions. Unlikely first error for those prompts. | `language_detector` tests. |
| LLM | Cannot be the first error if the transcript is already wrong. Mode B exists to isolate this. | issue #7 requirement |

### Fixes shipped

- Silero honours `min_speech_s`; default 0.25 s; preroll 0.25 s; hangover 0.7 s.
- Whisper: low-peak skip, hallucination drop, English retry, initial prompt, no previous-text conditioning.
- `test_audio.py` loads `config.yaml` (same mic + ASR).
- Structured `first_error_stage` on every turn.

### Recognition scores

`recognition.jsonl` is populated from the **2026-09-16 live ASR hypotheses** (audio not committed).

| Language | Items | Micro-WER | Mean item WER | Residual |
|---|---:|---:|---:|---|
| en | 10 | **0.048** | 0.050 | Turn 8 `Do you play sports?` → `2 play sports` |
| sa | 0 | — | — | no consented WAV |

#7’s sentence-recognition objective is **restated**: English WER on this set is ~0.05, not the blocking failure. `first_error_stage` is now `asr` for a leading-digit hypothesis; `looks_unusable` triggers English retry. If VAD ate “Do”, retry will still fail — accepted residual until a Mac re-measures. WER did not regress against the 0.06 quoted in #9.

## 4. Mode A — end-to-end simulator (spoken ×3)

**Status:** run 1 recorded (2026-09-16 offline). Runs 2–3 and all Bedrock/Pipecat spoken runs **not executed** in this worker (no daemon, no mic, 3.7 GiB). See `evals/results/mode_a_status.md`.

How to run on the operator Mac:

```text
# terminal 1
mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
# terminal 2 — Bedrock (Ollama must stay down)
python main.py --debug --llm-provider bedrock --llm-id us.anthropic.claude-sonnet-4-6
# say "Mitra", then each question three times
```

Repeat with `--llm-provider ollama` and with `--orchestrator pipecat`.

`scripts/eval_conversation.py --mode end-to-end --inject` (and `--orchestrator pipecat`) exercises wake-less **post-ASR** turns through both orchestrators + TTS using reference Sanskrit (fixture). That is **not** a substitute for spoken Mode A and is **not** counted as a result.

| Prompt | Run | Test mode | ASR | Model | Result |
|---|---:|---|---|---|---|
| (all ten) | 1 | mode-a-spoken-2026-09-16 | live | qwen3-vl:8b-instruct | **fail** gate 2.2/2.2 — `mode_a_run1_2026-09-16.md` |
| (all ten) | 2–3 | End-to-end spoken | *operator* | Qwen | **not run** (Mac) |
| (all ten) | 1–3 | End-to-end spoken | *operator* | Sonnet 4.6 | **not run** (Mac) |
| (all ten) | 1 | end-to-end-inject-custom | expected transcript | fixture reference | TTS path only — not a result |
| (all ten) | 1 | end-to-end-inject-pipecat | expected transcript | fixture reference | TTS path only — not a result |

## 5. Mode B — controlled LLM comparison

Identical prompts: `[lang=en] <expected>` + `SANSKRIT_SYSTEM_PROMPT`. No mic/VAD/ASR.

```bash
python scripts/eval_conversation.py --mode controlled --provider bedrock \
  --model-id us.amazon.nova-pro-v1:0
python scripts/eval_conversation.py --mode controlled --provider bedrock \
  --model-id us.anthropic.claude-sonnet-4-6
python scripts/eval_conversation.py --mode controlled --provider bedrock \
  --model-id us.openai.gpt-5.6-sol --max-tokens 512 --timeout-s 90
python scripts/eval_conversation.py --mode controlled --provider ollama \
  --model-id qwen3-vl:8b-instruct
```

**This worker (2026-09-08 retest after InvokeModel + inference-profile grant):**

| Model | Devanagari | Grammar mean | Semantic mean | Hard fail | Quality gate | Latency (s, n=10) |
|---|---|---:|---:|---|---|---|
| `us.amazon.nova-pro-v1:0` | 10/10 | 3.4 | 4.3 | yes (`नाश्नामी`) | **fail** | 0.55–0.92 |
| `us.anthropic.claude-sonnet-4-6` | 10/10 | 4.5 | 4.8 | no | **pass** | 1.46–3.09 |
| `us.anthropic.claude-haiku-4-5-20251001-v1:0` | 9/10 | 3.6 | 4.4 | yes (Hindi `खेल`, English gloss) | **fail** | 0.85–2.42 |
| `us.openai.gpt-5.6-sol` | 10/10 | 4.6 | 4.8 | no | **pass** | 3.4–23.5 (rejects `temperature`) |
| `us.anthropic.claude-sonnet-4-20250514-v1:0` | — | — | — | — | not run | legacy / ResourceNotFound |
| `qwen3-vl:8b-instruct` | 10/10 script | 2.2 | 2.2 | yes | **fail** (Mode A spoken, not Mode B) | no Ollama here; scored from 2026-09-16 log |

Scored tables: `conversation_nova_pro.md`, `conversation_sonnet46.md`, `conversation_haiku45.md`, `conversation_gpt56_sol.md`. JSONL + `mode_b_linguistic_scores.json` / `mode_b_aggregate.json`. Probe: `bedrock_probe.json`.

Ollama was **not** contacted (`ollama_contacted: false` on every Bedrock row). Failures still raise `ProviderError` with no silent swap.

**Do not** infer that Bedrock “fixes recognition.” Recognition is Mode A only.

## 6. Sanskrit quality gate

Scale 1–5 (issue table). Pass: all successful turns Devanagari; mean grammar ≥ 4.0; mean semantic ≥ 4.0; no score &lt; 3; gloss must not contradict.

### 6.1 Cloud-agent reference (not a candidate)

The cloud agent produced and scored target-quality replies (`src/eval/sanskrit_reference.py`). These demonstrate the rubric and a passing set. They are **not** Qwen or Bedrock generations.

| Prompt | Sanskrit | Grammar | Semantic | Gloss matches | Notes |
|---|---|---:|---:|---|---|
| What are you doing? | अहं त्वया सह वदामि। | 5 | 5 | yes | 1sg + सह + √वद् present |
| Where do you live? | अहं अत्र वसामि। | 5 | 4 | yes | Honest “here”; √वस् |
| Do you play? | आम्, अहं क्रीडामि। | 5 | 5 | yes | √क्रीड् |
| What’s your favorite food? | मम प्रियं भोजनं सेवफलम् अस्ति। | 4 | 4 | yes | Playful; lexicon apple; **uncertain / human review** |
| What’s your favorite subject? | मम प्रियः विषयः संस्कृतम् अस्ति। | 4 | 5 | yes | m/m agree; neuter language-name predicate |
| What are you reading? | अहं इदानीं न किञ्चित् पठामि। | 5 | 5 | yes | न not नहीं |
| Do you listen to music? | आम्, अहं संगीतं शृणोमि। | 5 | 5 | yes | acc + √श्रु |
| Do you play sports? | न, अहं क्रीडां न क्रीडामि। अहं मित्रम् अस्मि। | 4 | 4 | yes | Honest robot; slightly heavy negation |
| What will you do today? | अद्य अहं त्वया सह वदिष्यामि। | 5 | 5 | yes | future √वद् |
| Will you be my friend? | आम्, अहं तव मित्रम् अस्मि। | 5 | 5 | yes | तव मित्रम् |

Means: grammar **4.7**, semantic **4.7**. No score below 3. All Devanagari. Gate **passes for this reference set only**.

Corrected Sanskrit is omitted where score ≥ 4. Food item flagged uncertain (robots do not eat).

### 6.2 Candidate models (live Mode B)

| Model | Gate | Reason |
|---|---|---|
| Qwen3-VL 8B Instruct | **Fail** (Mode A spoken 2026-09-16) | Grammar 2.2 / semantic 2.2; Hindi; greeting template. Written Mode B still not run. |
| Nova Pro (`us.amazon.nova-pro-v1:0`) | **Fail** | Mean grammar 3.4; hard fail on `नाश्नामी`; several 3sg/gender errors |
| Claude Sonnet 4.6 (`us.anthropic.claude-sonnet-4-6`) | **Pass** | Mean grammar 4.5 / semantic 4.8; all Devanagari; lowest grammar 3 (`प्रिया विषयः`) |
| GPT-5.6 Sol (`us.openai.gpt-5.6-sol`) | **Pass** | Mean grammar 4.6 / semantic 4.8; all Devanagari; no score below 4. Converse rejects `temperature` (Mode B retries without it). |
| Claude Haiku 4.5 | **Fail** | Hindi `खेल` + English parenthetical on “Do you play?”; mean grammar 3.6 |
| Claude Sonnet 4 (20250514) | **Not scored** | Provider marks the ID legacy |

Sonnet 4.6 and GPT-5.6 Sol both passed the written gate on **prompt v1**. Nova Pro (grammar 3.4) and Haiku 4.5 (grammar 3.6, not all-Devanagari) **failed** — surface these negatives in the PR, not only in JSONL.

Offline Qwen **failed** the gate on live Mode A. The default-mode decision is recorded in ADR-001: `config.yaml` stays ollama for privacy; the recommended quality-gate mode is Bedrock Sonnet 4.6 + prompt v1. Prompt v2 was evaluated on Sonnet only and **regressed** the sports turn; it is not claimed as a Qwen fix.

The pipeline still *requires* Devanagari + TTS on every normal turn. `validation_ok` is script-only; `quality_ok` is the linguistic signal.

## 7. Vision and tools

Corpus: `evals/corpus/vision.yaml`.  
Script: `scripts/eval_vision.py --image-dir evals/fixtures/vision`.  
Images: **synthetic stand-ins** (identical across candidates), not MuJoCo captures. See `evals/results/vision.md`.

| Object | Sonnet 4.6 | Nova Pro |
|---|---|---|
| apple | सेवफलम् (grounded) | सेवफलम् (grounded) |
| croissant | पिष्टकः (pastry, partial) | “fan” / क्रितान्का (wrong) |
| duck | lexicon → कारण्डवः | lexicon → कारण्डवःो (broken) |

`ollama_loaded: false`. Qwen vision not run. `capture_image` tool loop still needs the simulator. Lexicon apple override remains unit-tested.

## 8. Pipecat

| Check | Result |
|---|---|
| Config flag | `orchestration.engine` / `--orchestrator` |
| Custom engine still default | Yes |
| Wake / barge-in / validate / lexicon via `handle_event` | Unit-tested on `PipecatOrchestrator` |
| Audio pump concurrency | Queues events; does not call `handle_event` on the mic thread |
| Ten-scenario inject (custom + Pipecat) | Unit-tested; TTS path + Sanskrit rubric scores recorded |
| Spoken MuJoCo e2e | Not run (Mac) |
| Echo gate / barge-in | Unit-tested custom + Pipecat |
| Recommendation | **Do not replace** the custom orchestrator (ADR-001) |

## 9. Bedrock mode checklist

| Requirement | Status |
|---|---|
| Text + captured images | Provider + `bedrock_converse` image path |
| Configurable id/region/timeout/retries/temp | Yes |
| Credential chain, no secrets in git | Yes |
| No Ollama contact in bedrock mode | `make_model` imports only Bedrock; `main.py --check` skips Ollama |
| Local Ollama remains | Default provider |
| Actionable errors, no silent swap | `ProviderError`; fallback default off |
| Raw audio local | ASR/TTS unchanged |

## 10. Recommendations

| Component | Choice | Evidence type |
|---|---|---|
| Orchestrator | **Custom** (keep Pipecat flag) | Tests + architecture; no live Pipecat-vs-custom latency |
| Bedrock LLM | **GPT-5.6 Sol** or **Claude Sonnet 4.6** for quality (both passed written gate); **Nova Pro** remains the cheap VLM shortlist but failed | Live Mode B on this worker |
| Offline LLM | **Qwen3-VL 8B Instruct** | Existing design + README baseline |
| Default mode | **config.yaml = ollama** (privacy). **Recommended quality-gate mode = Bedrock Sonnet 4.6 + prompt v1.** Offline Qwen failed Mode A run 1. | Mode A run 1 + Mode B |
| ASR | Whisper large-v3-turbo MLX + new guards | Code diagnosis |
| VAD / wake | Silero (fixed) / transcript “mitra” | Code diagnosis |
| TTS | Indic Parler-TTS, VITS fallback | Unchanged |
| On Bedrock provider failure | Surface `ProviderError`; optional explicit fallback | Config |
| Cloud ASR/TTS | Do not move | Privacy |

## 11. Unresolved / follow-up

1. ~~Grant `bedrock:InvokeModel` / inference-profile access and rerun Mode B.~~ Done 2026-09-08.
2. ~~Score the Qwen baseline.~~ Done from Mode A run 1 (`conversation_qwen.md`). Written Mode B Qwen still needs Ollama.
3. ~~Populate recognition WER.~~ Done (0.048 English micro-WER from the live log).
4. ~~Close the ADR default-mode deferral.~~ Done — offline fails the gate; recommended quality mode is Bedrock Sonnet 4.6 + prompt v1.
5. On the M1 Max: spoken Mode A **runs 2–3**, Bedrock spoken ×3, Pipecat spoken; attach `logs/turns.jsonl` (no audio).
6. Consented WAVs if a second WER (MLX vs transformers) is required.
7. Replace synthetic vision JPEGs with identical MuJoCo frames; run Qwen on the same files.
8. Human Sanskrit review of food/sports items and any score &lt; 5.
9. Custom openWakeWord “mitra” model (Phase 1).
10. Confirm Bedrock image+tool loop through Strands on the operator account.
11. Prompt v2 × Qwen Mode B (needs RAM) before claiming a local-model lift.

## 12. How to reproduce

```bash
cd mitra
python -m pytest tests -q --ignore=tests/hw
python scripts/eval_baseline.py
python scripts/eval_bedrock_probe.py
python scripts/eval_conversation.py --mode end-to-end --inject
python scripts/eval_conversation.py --mode end-to-end --inject --orchestrator pipecat
python scripts/eval_recognition.py          # corpus only
# with credentials and models enabled:
python scripts/eval_conversation.py --mode controlled --provider bedrock \
  --model-id us.amazon.nova-pro-v1:0
python scripts/eval_conversation.py --mode controlled --provider bedrock \
  --model-id us.anthropic.claude-sonnet-4-6
python scripts/eval_conversation.py --mode controlled --provider bedrock \
  --model-id us.openai.gpt-5.6-sol --max-tokens 512 --timeout-s 90
python scripts/eval_conversation.py --mode controlled --provider ollama \
  --model-id qwen3-vl:8b-instruct
```

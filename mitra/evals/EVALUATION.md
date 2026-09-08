# Evaluation report — issue #7

**Baseline commit (origin/main):** `c4dea80f8334bbcb31560b52ea8eb2de3d437dcf`  
**Branch:** `cursor/agent-MITRA-pipecat-cloudLLM-36fb`  
**Evaluator (Sanskrit rubric):** `cursor-grok-4.6-high-fast` (cloud agent). Not a candidate model.  
**PDF `tests/mitra-2026-08-22-1038-mobile.pdf`:** not present, not committed, not required. Corpus is `evals/corpus/conversation.yaml`.

## 1. Environment limits (measurements vs assumptions)

| Check | Result |
|---|---|
| Git baseline recorded | Yes — `evals/results/baseline.json` |
| Live Reachy Mini MuJoCo daemon | **No** — not installed / no display in this worker |
| Operator microphone | **No** |
| Local Ollama / `qwen3-vl:8b-instruct` | **No** |
| AWS identity | Assumed role `cursor_worker-execution-role`, `us-west-2` (account `146666888814`) |
| Worker RAM | **3.7 GiB** — local Qwen3-VL 8B cannot be loaded here |
| `bedrock:Converse` / `InvokeModel` | **AccessDenied** on Nova Micro/Lite/Pro/Premier, Claude Haiku 3, Claude Sonnet 4, Claude Haiku 4.5, Claude Sonnet 4.6, Llama 3 8B/70B, Mistral 7B (both `us-west-2` and `us-east-1`). See `evals/results/bedrock_probe.json`. |
| `bedrock:ListFoundationModels` | AccessDenied |
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

No consented WAVs are in git. `scripts/eval_recognition.py --audio-dir <dir>` computes WER/CER with the **application** Transcriber.

Until clips exist, WER is **not claimed**. Exact utterances that reproduce the bug on a Mac: the ten English questions in `evals/corpus/recognition.yaml`, spoken after “Mitra.”

## 4. Mode A — end-to-end simulator (spoken ×3)

**Status:** not executed in this worker (no daemon, no mic).

How to run on the operator Mac:

```text
# terminal 1
mjpython -m reachy_mini.daemon.app.main --sim --scene minimal
# terminal 2 — Bedrock (Ollama must stay down)
python main.py --debug --llm-provider bedrock --llm-id us.amazon.nova-pro-v1:0
# say "Mitra", then each question three times
```

Repeat with `--llm-provider ollama` and with `--orchestrator pipecat`.

`scripts/eval_conversation.py --mode end-to-end --inject` exercises wake-less **post-ASR** turns through `Orchestrator` + TTS using reference Sanskrit (fixture). That is **not** a substitute for spoken Mode A.

| Prompt | Run | Test mode | ASR | Model | Result |
|---|---:|---|---|---|---|
| (all ten) | 1–3 | End-to-end spoken | *operator* | Qwen / Nova / Sonnet | **pending Mac** |
| (all ten) | 1 | end-to-end-inject | expected transcript | fixture reference | TTS path unit-tested |

## 5. Mode B — controlled LLM comparison

Identical prompts: `[lang=en] <expected>` + `SANSKRIT_SYSTEM_PROMPT`. No mic/VAD/ASR.

```bash
python scripts/eval_conversation.py --mode controlled --provider bedrock \
  --model-id us.amazon.nova-pro-v1:0
python scripts/eval_conversation.py --mode controlled --provider bedrock \
  --model-id us.anthropic.claude-sonnet-4-20250514-v1:0
python scripts/eval_conversation.py --mode controlled --provider ollama \
  --model-id qwen3-vl:8b-instruct
```

**This worker:**

| Prompt | Run | Mode | Model | Sanskrit | Result |
|---|---:|---|---|---|---|
| What are you doing? | 1 | Controlled | `us.amazon.nova-pro-v1:0` | — | `access_denied` (no silent fallback) |
| (same set) | 1 | Controlled | `us.anthropic.claude-sonnet-4-20250514-v1:0` | — | `access_denied` |
| (same set) | 1 | Controlled | `qwen3-vl:8b-instruct` | — | strands-agents / Ollama missing; 3.7 GiB RAM cannot load 8B |

JSONL: `conversation_nova_pro.jsonl`, `conversation_sonnet4.jsonl`, `conversation_qwen.jsonl`. Probe: `bedrock_probe.json`.

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

### 6.2 Candidate models

| Model | Gate | Reason |
|---|---|---|
| Qwen3-VL 8B Instruct | **Not scored** | No Ollama in worker |
| Nova Pro | **Not scored** | IAM deny |
| Claude Sonnet 4 | **Not scored** | IAM deny |

**Blocking limitation:** a Bedrock or Qwen default cannot be declared to have passed the Sanskrit gate until Mode B is run with access. The pipeline will still *require* Devanagari + TTS on every normal turn.

## 7. Vision and tools

Corpus: `evals/corpus/vision.yaml` (apple / croissant / duck, MuJoCo `minimal`).  
Script: `scripts/eval_vision.py --image-dir …` (images not in git).

Lexicon override for **apple → सेवफलम्** is unit-tested (`test_verified_lexicon_overrides_generated_name`). Live VLM comparison is pending identical JPEGs + Bedrock access.

## 8. Pipecat

| Check | Result |
|---|---|
| Config flag | `orchestration.engine` / `--orchestrator` |
| Custom engine still default | Yes |
| Wake / barge-in / validate / lexicon via `handle_event` | Unit-tested on `PipecatOrchestrator` |
| Audio pump concurrency | Queues events; does not call `handle_event` on the mic thread |
| Spoken MuJoCo e2e | Pending Mac |
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
| Bedrock LLM | **Nova Pro** for cost, **Claude Sonnet 4** for quality — pick after Mode B | Research + IAM-blocked invoke |
| Offline LLM | **Qwen3-VL 8B Instruct** | Existing design + README baseline |
| Default mode | **ollama** until Sanskrit gate is measured on Bedrock | Constraint + missing live scores |
| ASR | Whisper large-v3-turbo MLX + new guards | Code diagnosis |
| VAD / wake | Silero (fixed) / transcript “mitra” | Code diagnosis |
| TTS | Indic Parler-TTS, VITS fallback | Unchanged |
| On Bedrock provider failure | Surface `ProviderError`; optional explicit fallback | Config |
| Cloud ASR/TTS | Do not move | Privacy |

## 11. Unresolved / follow-up

1. Grant `bedrock:InvokeModel` (and Converse) on the shortlist IDs; rerun Mode B; fill the results table.
2. On the M1 Max: spoken Mode A ×3 per question, both orchestrators, both providers; attach `logs/turns.jsonl` (no audio).
3. Consented recognition WAVs → WER/CER by language.
4. Identical simulator JPEGs → vision bake-off.
5. Human Sanskrit review of the food/sports items and any candidate score &lt; 5.
6. Custom openWakeWord “mitra” model (Phase 1).
7. Confirm Bedrock image+tool loop through Strands on the operator account.

## 12. How to reproduce

```bash
cd mitra
python -m pytest tests -q --ignore=tests/hw
python scripts/eval_baseline.py
python scripts/eval_bedrock_probe.py
python scripts/eval_conversation.py --mode end-to-end --inject
python scripts/eval_recognition.py          # corpus only
# with credentials and models enabled:
python scripts/eval_conversation.py --mode controlled --provider bedrock \
  --model-id us.amazon.nova-pro-v1:0
python scripts/eval_conversation.py --mode controlled --provider ollama \
  --model-id qwen3-vl:8b-instruct
```

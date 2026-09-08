# Model and service research matrix

**Date:** 2026-09-08  
**Scope:** GitHub issue #7 — Bedrock LLM/VLM, ASR, VAD/wake, TTS.  
**Privacy boundary (unchanged):** raw microphone audio stays on the host. Only transcribed text and explicitly captured images may leave the machine.

Sources are linked per row. “Newest/largest” was not used as a selection rule.

## 1. LLM / VLM on Amazon Bedrock

| Candidate | Exact ID (typical) | Region notes | Multilingual | Sanskrit / Devanagari | Vision | Tools | Streaming | Context | Latency / cost (public) | Laptop load | Decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Amazon Nova Pro | `amazon.nova-pro-v1:0` / inference profile `us.amazon.nova-pro-v1:0` | On-demand in `us-east-1`; CRIS includes `us-west-2` ([Nova UG](https://docs.aws.amazon.com/nova/latest/userguide/what-is-nova.html)) | 200+ listed; optimized 15 do **not** include Sanskrit | Mode B 2026-09-08: all Devanagari; grammar **3.4** / semantic **4.3**; hard fail `नाश्नामी`. Hindi is optimized; Sanskrit is not claimed | Text, image, video | Converse tool use | Yes | 300k in / 10k out | Lowest $ among capable VLMs; this worker ~0.6–0.9 s | None (cloud) | **Shortlist A (cost/VLM)** — written Sanskrit gate **failed**. Keep for vision A/B, not as default. |
| Amazon Nova Lite | `amazon.nova-lite-v1:0` / `us.amazon.nova-lite-v1:0` | Same CRIS story as Pro | Same as Pro | Weaker instruction following expected | Yes | Yes | Yes | 300k | Cheaper/faster than Pro | None | **Secondary** — good A/B for latency; risk on grammar. |
| Amazon Nova Premier | `amazon.nova-premier-v1:0` / `us.amazon.nova-premier-v1:0` | `us-east-1` + CRIS | Same | Unknown | Yes | Yes | Yes | 1M | Highest Nova cost | None | Rejected as default: cost/size without evidence of better Sanskrit. |
| Amazon Nova Micro | `amazon.nova-micro-v1:0` | Same family | Text only | n/a | **No** | Yes | Yes | 128k | Very cheap | None | **Rejected** — no vision (FR-2). |
| Amazon Nova Sonic | `amazon.nova-sonic-v1:0` | `us-east-1`, `eu-north-1`, `ap-northeast-1` | EN/FR/IT/DE/ES speech | No Sanskrit speech | Speech I/O | Tools via speech | Bidirectional | 300k | Speech-native | None | **Rejected** — would send raw audio off-host; no Sanskrit TTS. |
| Amazon Nova 2 Lite | `amazon.nova-2-lite-v1:0` / `us.amazon.nova-2-lite-v1:0` / `global.amazon.nova-2-lite-v1:0` | [Nova 2 inference](https://docs.aws.amazon.com/nova/latest/nova2-userguide/core-inference.html) | Multimodal | Unknown | Yes (text/image/video/audio) | Yes | Yes | large | Newer SKU | None | Watch-list. Not selected solely for being newer. Audio input unused (privacy). |
| Claude Sonnet 4 | `anthropic.claude-sonnet-4-20250514-v1:0` / `us.anthropic.claude-sonnet-4-20250514-v1:0` | Cross-region profiles; confirm console | Strong Indic script track record on prior Claude | Expected best Sanskrit among shortlist; **must measure** | Images | Best-in-class tools | Yes | 200k class | Higher $ than Nova Pro | None | **Shortlist B** — quality/tool candidate. |
| Claude Sonnet 4.6 | `anthropic.claude-sonnet-4-6` / `us.anthropic.claude-sonnet-4-6` / `global.anthropic.claude-sonnet-4-6` | [model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-sonnet-4-6.html); geo from `us-west-2`. Bare `anthropic.claude-sonnet-4-6` needs an inference profile. | Strong | Mode B 2026-09-08: grammar **4.5** / semantic **4.8**; gate **pass** | Images | Yes | Yes | 200k class | This worker ~1.5–3.1 s | None | **Quality pick** among live IDs. Sonnet 4 (20250514) is legacy. |
| Claude Haiku 4.5 | `anthropic.claude-haiku-4-5-20251001-v1:0` / `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Profile IDs vary by Region | Strong | Mode B 2026-09-08: grammar **3.6**; Hindi `खेल` + English on “Do you play?”; gate **fail** | Images | Yes | Yes | 200k class | This worker ~0.9–2.4 s | None | Faster than Sonnet; **not** a Sanskrit default. |
| Claude 3.5 Sonnet / Haiku (2024 ids) | `anthropic.claude-3-5-sonnet-20241022-v2:0` etc. | — | — | — | — | — | — | — | — | — | **Rejected 2026-09-08:** `ResourceNotFoundException` — “model version has reached the end of its life.” |
| Llama 4 Scout / Maverick | `meta.llama4-scout-17b-instruct-v1:0`, `meta.llama4-maverick-17b-instruct-v1:0` | US CRIS | Multilingual | Unknown Devanagari quality | Scout/Maverick are multimodal | Tool support uneven vs Claude/Nova | Yes | large | Mid | None | Backup only; tool+vision maturity less operationally proven in Strands. |
| Qwen on Bedrock (if enabled) | account-specific | Region-specific | Native Chinese/EN; some Indic | Same family as local baseline | Some SKUs | Varies | Yes | varies | Mid | None | Interesting like-for-like vs local Qwen; **do not assume access**. |
| GPT-5.6 Sol | `openai.gpt-5.6-sol` / `us.openai.gpt-5.6-sol` / `global.openai.gpt-5.6-sol` | [model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-sol.html). `bedrock-runtime` needs a CRIS profile. | Strong | Mode B 2026-09-08: grammar **4.6** / semantic **4.8**; gate **pass**. Rejects `temperature`; `maxTokens` ≥ 16. | Text, image | Tools via Responses; Converse works | Yes | 1M | Geo CRIS ~$4.40 / $22 per 1M; this worker 3–24 s | None (cloud) | **Quality pick** with Sonnet 4.6. Do not send `temperature`. |
| Local Qwen3-VL 8B Instruct | `qwen3-vl:8b-instruct` via Ollama | n/a | EN + vision; Sanskrit via prompt | Current baseline; validator often needed | Yes | Native Ollama tools | No (local HTTP) | ~32k–128k class | $0 runtime; ~6 GB RAM; ~3 s warm on M1 Max (README) | **High** | **Offline default.** Not loaded when `provider: bedrock`. |

### Rejected / out of policy

| Candidate | Reason |
|---|---|
| Any cloud ASR (Transcribe, Whisper API, Nova Sonic input) | Raw audio would leave the host. Needs an explicit privacy decision (issue out of scope). |
| Hard-coded model IDs as the only option | Operators must set `models.llm.id` and Region. |
| Silent fallback Ollama on Bedrock errors | Disabled unless `models.llm.fallback.enabled: true`. |

### Bedrock wiring (this repo)

- Credentials: standard AWS chain only (`boto3.Session().get_credentials()`).
- Config: `models.llm.provider`, `id`, `region`, `temperature`, `timeout_s`, `max_retries`, `max_tokens`, `streaming`.
- Strands `BedrockModel` ([docs](https://strandsagents.com/docs/user-guide/concepts/model-providers/amazon-bedrock/)) plus `mitra.eval.bedrock_converse` for Mode B without the agent extra.

**This worker’s IAM** (`cursor_worker-execution-role` in `us-west-2`) can `Converse` and `InvokeModel` on the inference-profile IDs above (retest 2026-09-08). Bare foundation-model IDs are rejected (`ValidationException`: use an inference profile). `ListFoundationModels` is still denied and is not required to invoke. Probe: `python scripts/eval_bedrock_probe.py` → `evals/results/bedrock_probe.json`.

## 2. ASR (must stay local)

| Candidate | ID | Why considered | Decision |
|---|---|---|---|
| Whisper large-v3-turbo (MLX) | `mlx-community/whisper-large-v3-turbo` | Current default; en/kn; fast on Apple Silicon | **Keep.** Added hallucination filter, `condition_on_previous_text=false`, low-peak skip, English retry. |
| Whisper small (wake only) | `mlx-community/whisper-small-mlx` | Already used for wake; tiny mis-hears “mitra” | **Keep for wake.** |
| Sanskrit Whisper fine-tune | `models.asr.sanskrit` (unset) | Experimental rescue for Devanagari | Keep hook; no model selected (R7). |
| Amazon Transcribe / Whisper API | various | Higher en accuracy possible | **Rejected** — raw audio off-host. |
| `faster-whisper` / whisper.cpp | local | Portability off macOS | Future; v1 stays MLX. |

## 3. VAD and wake

| Candidate | ID | Decision |
|---|---|---|
| Silero VAD | `silero` + `VADIterator` | **Keep.** Bugfix: `min_speech_s` was dropped for Silero; short bursts became utterances. Now enforced; `min_silence_s` 0.7 s; preroll 0.25 s. |
| Energy VAD | `energy` | Fallback when torch/silero missing; adaptive floor. |
| Transcript wake (“mitra”) | whisper-small match | **Keep** until custom openWakeWord ONNX exists (Phase 1). |
| openWakeWord custom | `models/mitra.onnx` | Target, not ready. |
| Porcupine | — | Not fully open-source; only if openWakeWord misses FR-1.4 (REQUIREMENTS). |

## 4. TTS (Sanskrit / Kannada / English)

| Candidate | ID | Sanskrit | Decision |
|---|---|---|---|
| Indic Parler-TTS | `ai4bharat/indic-parler-tts` | Best current local Sanskrit | **Keep primary.** |
| MMS VITS Hindi | `facebook/mms-tts-hin` | Approximate Sanskrit | **Keep fallback** (already wired). |
| Amazon Polly (Aditi/Kajal etc.) | Polly engine | Hindi/English, not classical Sanskrit | Rejected without a privacy+quality review. |
| Nova Sonic output | speech model | No Sanskrit | Rejected. |
| macOS `say` | `Junior` etc. | English only | Used only by `tests/mini_conversation_app.py`, not Mitra. |

## 5. Orchestration

| Candidate | Replaces | Decision |
|---|---|---|
| Custom `Orchestrator` | — | **Keep default.** Measured, deterministic nod/validate/speak. |
| Pipecat PoC | Audio pump / dispatch only | **Retain behind flag.** Does not replace validator, lexicon, tools, local ASR/TTS. Daily WebRTC/cloud STT not adopted (privacy). See ADR-001. |

## 6. Cost sketch (Bedrock, public list, USD, order-of-magnitude)

Per short Mitra turn (~400 input tokens prompt+history, ~80 output tokens), **without** images:

| Model | Rough USD / turn | 10 questions × 3 runs |
|---|---|---|
| Nova Lite | well under $0.01 | under $0.05 |
| Nova Pro | ~$0.01 | < $0.15 |
| Claude Sonnet 4 | ~$0.02–0.05 | < $0.50 |
| Vision turn (one JPEG) | + image tokens | budget separately |

Exact rates change; confirm [Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) at run time. Laptop memory in Bedrock mode should drop ~6 GB versus resident Qwen.

## 7. What was not changed

- Privacy boundary for microphone audio.
- Default `config.yaml` provider remains `ollama` / `qwen3-vl:8b-instruct` so offline mode stays explicit and tests stay stable.
- Deterministic validator and verified lexicon.

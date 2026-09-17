# ASR, TTS, and memory (issue #9 §11)

## ASR

Live 2026-09-16: `whisper-small` on the transformers backend, **not** MLX. Warm ASR ≈ 3.7 s of a 5.5 s turn.

| Backend | Runnable here | Equal-accuracy / lower-latency claim |
|---|---|---|
| MLX whisper-large-v3-turbo (config default) | No — Linux ARM, no MLX | Last operator baseline on M1 Max. Not re-measured. |
| transformers whisper-small (the 2026-09-16 run) | Not loaded in this 3.7 GiB worker | WER 0.048 on the ten English items; one Do→2 error. |
| openai-whisper / faster-whisper / whisper.cpp | Optional `mitra[asr-cpu]` | Not benchmarked; no consented WAVs to decode. |

`Do you play sports?` → `2 play sports` is treated as unusable (leading digit) so English retry fires. If VAD truncated “Do”, retry will still fail and `first_error_stage=asr`. Accepted residual until a Mac re-measures with preroll.

Recognition objective from #7: **restated**. Sentence recognition on this English set is largely fine (micro-WER 0.048). The blocking failure is Sanskrit response quality.

## TTS

Live 2026-09-16 used `facebook/mms-tts-hin` (Hindi VITS fallback). Spoken Sanskrit is currently rendered by a Hindi model.

| Voice | Runnable here | Note |
|---|---|---|
| Indic Parler-TTS (`ai4bharat/indic-parler-tts`) | No (download + RAM) | Config primary; Sanskrit-appropriate target. |
| `facebook/mms-tts-hin` | Not loaded | Fallback actually used on 2026-09-16. Hindi phonology. |

A pronunciation bake-off needs the operator Mac. Do not move TTS off-host.

## Memory

| Session | Turns | RSS start → end | Δ |
|---|---:|---|---|
| 2026-09-16 live (Whisper+TTS resident, Ollama excluded) | 10 | 4120 → 5070 MB | **+950 MB** |
| Inject + FakeReachy + canned agent (this worker) | 30 | 30.9 → 31.5 MB | +0.6 MB |

The orchestrator/history path does not leak at the 950 MB scale. The live growth is model caches (ASR/TTS) and/or allocator fragmentation, not conversation history. A 30-turn **model-resident** curve still needs the Mac.

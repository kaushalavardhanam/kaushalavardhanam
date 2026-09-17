# Pipecat end-to-end (issue #9 §9)

Spoken MuJoCo + mic: **not run** (no daemon, no display, no operator).

What was verified in this PR:

| Check | Custom | Pipecat |
|---|---|---|
| Wake / barge-in via `handle_event` | unit test | unit test |
| Ten-scenario inject + TTS | unit test + `eval_conversation --inject` | same |
| Echo gate suppresses playback transcription | unit test | `WakeGateProcessor` + `VadProcessor` |
| Echo gate still allows high-RMS barge-in | unit test | unit test |
| Conversation history / Bedrock / `capture_image` live | not run | not run |
| Session reset | existing orchestrator tests | inherits `handle_event` |

**Blockers for replacing the custom orchestrator:** no spoken A/B latency, no live Bedrock+tools loop on this path, Daily transports still violate the privacy boundary. **Recommendation unchanged:** keep `orchestration.engine: custom`; retain the PoC flag.

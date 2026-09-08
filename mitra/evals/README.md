# Mitra evaluation (issue #7)

| Doc | Contents |
|---|---|
| [MODEL_RESEARCH.md](MODEL_RESEARCH.md) | Bedrock / ASR / VAD / TTS comparison matrix |
| [ADR-001-pipecat-bedrock.md](ADR-001-pipecat-bedrock.md) | Keep custom orchestrator; Bedrock is explicit |
| [EVALUATION.md](EVALUATION.md) | Baseline, recognition diagnosis, Mode A/B, Sanskrit gate |
| [corpus/](corpus/) | Conversation, recognition, vision scenarios (no PDF, no private audio) |
| [results/](results/) | Baseline snapshot and script outputs |

```bash
python scripts/eval_baseline.py
python scripts/eval_bedrock_probe.py
python scripts/eval_recognition.py --audio-dir /consented/wavs
python scripts/eval_conversation.py --mode controlled --provider bedrock --model-id us.amazon.nova-pro-v1:0
python scripts/eval_conversation.py --mode controlled --provider bedrock --model-id us.anthropic.claude-sonnet-4-6
python scripts/eval_conversation.py --mode end-to-end --inject
```

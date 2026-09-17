# Mitra evaluation (issue #7 / #9)

| Doc | Contents |
|---|---|
| [MODEL_RESEARCH.md](MODEL_RESEARCH.md) | Bedrock / ASR / VAD / TTS comparison matrix |
| [ADR-001-pipecat-bedrock.md](ADR-001-pipecat-bedrock.md) | Default-mode decision; keep custom orchestrator |
| [EVALUATION.md](EVALUATION.md) | Mode A run 1, recognition WER, Mode B, Sanskrit gate |
| [corpus/](corpus/) | Conversation, recognition, vision scenarios (no PDF, no private audio) |
| [results/](results/) | Mode A run 1, scored Qwen, WER, vision, prompt ablation |
| [fixtures/vision/](fixtures/vision/) | Synthetic identical JPEGs (not MuJoCo) |

```bash
python scripts/eval_baseline.py
python scripts/eval_bedrock_probe.py
python scripts/eval_recognition.py --audio-dir /consented/wavs
python scripts/eval_conversation.py --mode controlled --provider bedrock --model-id us.amazon.nova-pro-v1:0
python scripts/eval_conversation.py --mode controlled --provider bedrock --model-id us.anthropic.claude-sonnet-4-6
python scripts/eval_conversation.py --mode controlled --provider bedrock --model-id us.openai.gpt-5.6-sol --max-tokens 512 --timeout-s 90
python scripts/eval_conversation.py --mode end-to-end --inject
python scripts/record_issue9_evidence.py
python scripts/eval_conversation.py --mode controlled --provider bedrock \
  --model-id us.anthropic.claude-sonnet-4-6 --prompt-version v2
python scripts/eval_vision.py --image-dir evals/fixtures/vision \
  --provider bedrock --model-id us.anthropic.claude-sonnet-4-6
```

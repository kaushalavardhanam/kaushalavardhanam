# Coherence Check: GPT5.6 Client (Issue #13, sub-task 4)

`mitra/coherence/judge_client.py` wraps GPT5.6 for three jobs:

1. **Follow-up turns** – `followup_reply(history)` returns the next Sanskrit
   reply after the first exchange, so dialogues can continue.
2. **Baseline** – `baseline_reply(history)` returns a plain GPT5.6 reply to the
   same prompt(s), for comparison with the Reachy Mini agent.
3. **Fallback scoring** – `score_metric(metric, text, history=None)` scores
   metrics that cannot be computed programmatically.

Note: written without access to the API; the request format follows the
OpenAI-compatible `/chat/completions` endpoint and has not been run against a
live account. Try `python -m coherence.judge_client baseline "..."` first.

## Configuration

Set `OPENAI_API_KEY` and `MITRA_JUDGE_MODEL` (see `coherence-check-setup.md`).
Optional: `OPENAI_BASE_URL`, and `MITRA_JUDGE_CACHE` for the cache directory
(default `mitra/logs/judge_cache/`, not to be committed).

## Usage (from `mitra/`)

```python
from coherence.judge_client import JudgeClient

client = JudgeClient.from_env()
history = [
    {"role": "user", "text": seed_question},
    {"role": "assistant", "text": agent_first_reply},
]
next_line = client.followup_reply(history, sample=run)
baseline = client.baseline_reply([history[0]], sample=run)
result = client.score_metric("coherence", next_line, history=history)
# {"score": 4, "rationale": "..."}
```

History roles are `user` and `assistant` (`agent` is an alias for
`assistant`).

## Structured prompts and rubric

- All calls request a JSON object: `{"reply": "..."}` for dialogue and
  `{"score": int, "rationale": "..."}` for scoring. Output is parsed
  defensively (code fences removed, one re-ask on invalid JSON).
- Scoring uses one fixed scale, integers 1–5, and per-metric rubrics defined in
  `RUBRICS` (`coherence`, `relevance`, `fluency`, `language_purity`). Add a
  metric by adding a rubric entry; unknown metrics raise an error.
- The scoring input is itself JSON (metric, rubric, scale, dialogue so far,
  text) so every metric is judged the same way.

## Temperature

| Call | Default |
| --- | --- |
| `followup_reply`, `baseline_reply` | 0.7 |
| `score_metric` | 0.0 |

If the model rejects a custom temperature, the client retries without it and
stops sending it for the rest of the session.

## Caching

Each request is cached on disk by a hash of the call kind, model, temperature,
messages and `sample`. Re-running the same harness costs no API calls. Use a
different `sample` (for example the run number) to get independent
generations for the same prompt. Delete the cache directory to regenerate.

Transient errors (429, 5xx, network) are retried with exponential backoff.
Failures raise `JudgeError`.
# Coherence Check: GPT5.6-Assisted Metrics (Issue #13, sub-task 6)

`mitra/eval/judged_metrics.py` uses the judge model for the metrics that cannot
be computed by code:

- **Cross-turn repeats / parroting** (`repeats`)
- **Hindi-influenced forms** (`hindi_forms`)
- **Any other metric without a programmatic calculation** (`custom:<name>`,
  given a rubric)

## How judging works

Each harness record is judged with the **previous turns of the same dialogue**
(same category, seed question and run, ordered by depth) in the prompt. Depth 0
gets an empty history. The judge must return one JSON object with integer
counts and quoted examples:

| Kind | Required keys |
| --- | --- |
| `repeats` | `repeat_count`, `parroting_count`, `examples[{type, span, earlier_turn, note}]` |
| `hindi_forms` | `hindi_form_count`, `examples[{span, hindi_form, sanskrit_alternative, note}]` |
| `custom:<name>` | `count`, optional `score`, `examples[{span, note}]` |

Malformed output (not JSON, missing keys, non-integer counts) is retried once
with the error appended, then stored as an `error` entry rather than dropped.
If a count disagrees with the number of examples, a `warnings` field is added
so you can look at it during the spot check.

## Usage (from `mitra/`)

```bash
python -m eval.judged_metrics run                      # repeats + hindi
python -m eval.judged_metrics run --metrics repeats
python -m eval.judged_metrics run --custom-metric register \
    --rubric "Count sentences that switch between formal and colloquial register."
python -m eval.judged_metrics summary                  # per kind/category/depth
```

Results append to `mitra/logs/judged_metrics.jsonl` (`MITRA_JUDGED_LOG`).
Already judged items are skipped on re-run; use `--force` to redo them.
Credentials: `OPENAI_API_KEY`, `MITRA_JUDGE_MODEL`, optional `OPENAI_BASE_URL`
(check with `python -m coherence.check_env`). The default judge calls an
OpenAI-compatible chat-completions endpoint directly; pass your own
`judge_fn(prompt) -> str` to the Python functions to use
`coherence/judge_client.py` instead.

## Manual spot check

```bash
python -m eval.judged_metrics sample --n 10   # fixed seed 13, prints turns + judge output
# edit mitra/logs/judged_spotcheck.jsonl: set manual_agree true/false, add manual_note
python -m eval.judged_metrics report          # agreement rate
```

Read each sampled turn with its history and check that the quoted examples
really appear and are correct. This step has not been run yet: it needs real
judge output from a simulator run. Record the agreement rate and any systematic
errors (e.g. Sanskrit words wrongly flagged as Hindi) in the issue before
trusting the numbers.
# Coherence Check: Test-Harness Logger (Issue #13, sub-task 2)

`mitra/eval/harness.py` records every Reachy Mini reply as a structured JSONL
record so per-category, per-depth scores can be recomputed later.

## Record fields

| Field | Meaning |
| --- | --- |
| `ts` | UTC ISO-8601 timestamp |
| `category` | Question category / component under test |
| `seed_question` | Seed question that started the dialogue |
| `run` | Run number |
| `depth` | Dialogue depth: first spoken line = 0, nth spoken line = n-1 |
| `text` | Raw reply text (UTF-8, Devanagari preserved) |
| `grammar_score` | Output of the grammar verifier (`null` if none supplied) |
| `hindi_flag` | Output of the Hindi detector (`null` if none supplied) |
| `verifier_error` | Present only if a verifier raised an exception |

## Usage

Run from the `mitra/` directory.

```python
from eval.harness import Harness

harness = Harness(grammar_fn=grammar_scorer, hindi_fn=hindi_detector)
conv = harness.conversation(category="greeting", seed_question="...", run=1)
conv.log_line(first_reply)    # depth 0
conv.log_line(second_reply)   # depth 1
```

`grammar_fn(text) -> float` and `hindi_fn(text) -> bool` are supplied by the
caller; the harness does not define the verifiers. Log every spoken line
(one `log_line` call per line) so depth stays accurate.

Records go to `mitra/logs/harness_records.jsonl`, or to the path in
`MITRA_HARNESS_LOG`. Do not commit log files.

## Recomputing scores

```bash
python -m eval.harness summary          # per category and depth: n, mean grammar, Hindi rate
python -m eval.harness csv results.csv  # export for spreadsheets / pandas
```

From Python, use `eval.harness.load_records()` and `summarize()`.
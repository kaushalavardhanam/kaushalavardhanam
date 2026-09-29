# Coherence Check: Automated Component Metrics (Issue #13, sub-task 5)

`mitra/eval/metrics.py` computes per-group metrics over Reachy Mini replies.

| Metric | Source |
| --- | --- |
| `grammar_score_mean` | Grammar verifier score (stored by the harness), non-empty replies only |
| `recognition_rate` | Share of Devanagari words found by a Vidyut lookup (pooled) |
| `pronoun_verb_errors` | Heuristic pronoun-verb person/number mismatches per sentence |
| `replies_with_hindi` | Hindi detector flag (stored by the harness) |
| `replies_with_latin` | Regex for Latin-script letters |
| `replies_with_4gram_repeat` | Replies where a word 4-gram occurs more than once |
| `median_words` | Median words per non-empty reply |
| `empty_replies` | Replies that are missing or whitespace only |

## Usage (from `mitra/`)

```bash
python -m eval.metrics                 # per-category table from harness records
python -m eval.metrics --vidyut        # also compute recognition rate
python -m unittest tests.test_metrics  # unit tests
```

From Python: `compute_metrics(replies, grammar_scores, hindi_flags, word_lookup)`
or `metrics_from_records(records, word_lookup)`.

## Caveats

- `--vidyut` needs `vidyut` installed and `MITRA_VIDYUT_DATA` pointing at the
  Vidyut data directory (kosha under `<dir>/kosha`). This was written without
  running against a real Vidyut install; check the `Kosha(...).contains` call
  against your version.
- Pronoun-verb agreement is an ending-based heuristic and can misfire on nouns
  that look like verb forms. Use it as a screening signal.
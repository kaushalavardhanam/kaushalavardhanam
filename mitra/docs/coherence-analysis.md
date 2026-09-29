# Coherence Check: Result Analysis and Improvement Suggestions (Issue #13, sub-task 11)

`mitra/eval/analysis.py` turns the harness records
(`mitra/logs/harness_records.jsonl`) into a ranked list of problems, depth
trends, a comparison with the GPT5.6 judge, and concrete recommendations.

```bash
cd mitra
python -m eval.analysis report                 # markdown report
python -m eval.analysis report --json          # machine-readable
python -m eval.analysis --path other.jsonl report
```

Note: this document and the tool were written without access to real
campaign results. No numbers are claimed here. Run the report on the
recorded data, paste it into the "Findings" section below, and keep only the
recommendations the data supports.

## What the report contains

1. **Issues ranked by ease of fix** – detected per reply from the text and
   the verifier flags, sorted easiest first, ties broken by prevalence.
2. **Components dragging quality down** – per category: mean grammar score,
   *drag* (overall mean minus category mean; positive = worse than average)
   and share of replies with any issue. Worst categories come first.
3. **Depth trend** – per depth mean grammar and issue rate, plus the
   least-squares slope over depth. A slope below -0.01 for grammar (or above
   +0.01 for issue rate) triggers a depth recommendation.
4. **Local verifier vs GPT5.6** – uses records that carry both
   `grammar_score` and `judge_score` (pass `judge_score=...` as an extra
   keyword to `Conversation.log_line`; it is assumed to be on the same 0..1
   scale). The gap is local minus judge.
5. **Recommendations** – generated from the rates above.

## Issue types and ease of fix

Thresholds (in `eval/analysis.py`): Latin share > 5% of letters, grammar
score < 0.5, repetition when distinct tokens < 50% of a reply of 6+ tokens.
These are heuristics; tune them if they misfire.

| Ease | Issue | Detection | Recommended fix |
| --- | --- | --- | --- |
| 1 (easiest) | Empty reply | blank text | Verifier-in-the-loop retry; fixed Sanskrit fallback line |
| 2 | Latin-script leakage | Latin letters > 5% | Logit bias / banned-token list against Latin tokens; filter and retry or transliterate |
| 3 | Repetition / looping | few distinct tokens | Repetition penalty, no-repeat-ngram, shorter max length, cut at first repeated sentence |
| 4 | Hindi instead of Sanskrit | `hindi_flag` | Prompt: Sanskrit only, never Hindi, with short Sanskrit example turns; Hindi detector as retry trigger |
| 5 (hardest) | Low grammar score | `grammar_score` < 0.5 | Rerank multiple samples with the grammar verifier, shorter replies, targeted few-shot examples, fine-tuning only as a last resort |

Rule of thumb: fix issues 1-3 first with filters and decoding constraints.
They need no model change, are cheap to verify with the existing harness, and
each removes a class of failures that pulls purity down regardless of
grammar quality. Issues 4-5 need prompt work or model work and should be
attacked afterwards, starting with the categories the report shows as the
largest drag.

## Recommendation menu

- **Prompt changes**
  - State the language rule ("reply only in Sanskrit, Devanagari script, no
    Hindi, no English") and repeat it in every turn, not only the system
    prompt, so it survives deep dialogues.
  - Add 1-2 short, correct Sanskrit example turns for the weakest categories.
  - Ask for short replies (one or two simple sentences); fewer clauses means
    fewer grammar errors.
- **Decoding constraints**
  - Penalize or ban Latin-script tokens.
  - Repetition penalty / no-repeat-ngram, and a sensible max token limit.
  - Lower temperature for categories with high variance across the 3 runs.
- **Filters**
  - Script filter (Latin share) and empty-reply filter applied before TTS.
  - Hindi detector as a gate before speaking.
- **Verifier-in-the-loop retries**
  - Generate, run the grammar verifier and the filters, resample up to N
    times (e.g. 3) if any check fails, and speak the best-scoring candidate.
  - Log the number of retries in the harness (extra field) so the cost in
    latency can be measured; on a CPU-lightweight setup this latency matters.
- **Depth-specific**
  - If quality falls with depth: re-inject the instruction each turn, trim or
    summarize history, and run retries on every turn.
- **Judge calibration**
  - If the local verifier and GPT5.6 disagree substantially, recalibrate the
    local threshold before using it as the retry trigger; use GPT5.6 offline
    for evaluation, not in the live loop.

## Findings (to fill in after running the report)

- Worst categories (largest drag): _TBD_
- Most frequent issues and their rates: _TBD_
- Depth trend (grammar slope, issue slope): _TBD_
- Differences from GPT5.6 (per-category gap, direction): _TBD_
- Recommendations adopted, in priority order: _TBD_

## Tests

```bash
cd mitra
python -m pytest tests/test_analysis.py
```
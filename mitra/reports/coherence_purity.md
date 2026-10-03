# MITRA Coherence and Purity Check: Final Report

Issue: #13 (agent-MITRA-CoherenceCheck-1)
Branch: `agent-mitra-coherence-check-1` (branched from `asr-cpu-lightweight`)

> **Status of this document.** This is the report skeleton. It was written
> without running the campaign or the analysis, so every value marked
> `TBD` must be filled in from the real outputs before the report is
> considered final. No numbers below have been measured. Do not
> publish the report while any `TBD` remains.

## 1. Summary of findings and suggestions

### Findings

Fill in from the tables, scatterplot, and regression in sections 3 to 5.

- **Overall grammar quality:** TBD (mean grammar score across all logged
  lines, and how it varies by category).
- **Language purity (Hindi leakage):** TBD (overall share of lines flagged as
  Hindi by the detector, and which categories are worst).
- **Effect of dialogue depth:** TBD (does the grammar score fall, or the Hindi
  rate rise, as the dialogue gets deeper? Give the direction and size from the
  regression in section 5).
- **Relationship between grammar score and Hindi flag:** TBD (state what the
  scatterplot and regression show, and whether it is statistically
  distinguishable from no relationship).
- **Judged coherence metrics:** TBD (summarise the judge-model results, see
  `mitra/docs/coherence-judged-metrics.md`).
- **Caveats:** TBD (sample size per cell, verifier limitations, judge-model
  variance, any runs that failed or were excluded).

### Suggestions

Fill in once the findings are known. Candidate directions, to keep only if
the data supports them:

- If Hindi leakage rises with depth: constrain the system prompt or add a
  language-check-and-retry step on each reply.
- If particular categories score poorly: add category-specific examples or
  vocabulary support for those categories.
- If the verifiers disagree with the judge model: review verifier thresholds
  before relying on them.

## 2. Methodology

Detailed notes for each stage are in `mitra/docs/`.

1. **Setup.** The agent runs in the Reachy Mini MuJoCo simulator on the
   `asr-cpu-lightweight` base. Each spoken line is logged
   (`mitra/docs/coherence-check-setup.md`).
2. **Conversation starters.** Categories and questions come from
   <https://sanskritdocuments.org/doc_z_misc_major_works/daily.html>. The
   parser looks for 27 categories. Up to 3 questions per category are
   sampled at random with a fixed seed (`SEED = 13`), and each question is
   tested 3 times (`RUNS_PER_QUESTION = 3`). The selection is stored in a
   versioned file (`mitra/docs/coherence-starters.md`). TBD: state the
   number of categories actually found and the number of questions used.
3. **Harness.** Every reply is stored as a structured record with category,
   seed question, run, depth, text, grammar score and Hindi flag. Depth is 0
   for the first spoken line and n-1 for the nth
   (`mitra/docs/coherence-harness.md`).
4. **Metrics and scoring.** Grammar score and Hindi flag come from the
   verifiers (`mitra/docs/coherence-metrics.md`,
   `mitra/docs/coherence-scoring.md`). Judge-model metrics use GPT5.6 through
   the judge client (`mitra/docs/coherence-judge-client.md`,
   `mitra/docs/coherence-judged-metrics.md`). TBD: record the exact judge
   model identifier used (`MITRA_JUDGE_MODEL`) and the date of the run.
5. **Campaign.** The campaign runner drives the dialogues and fills the
   harness log (`mitra/docs/coherence-campaign.md`). TBD: dialogue length in
   turns, and the total number of records collected.
6. **Analysis.** Two two-way tables, a scatterplot, and a regression are
   produced from the records (`mitra/docs/coherence-analysis.md`,
   `mitra/docs/coherence-tables.md`, `mitra/docs/coherence-regression.md`).

Known limitations to state honestly in the final version: the
starters-page parser is heuristic; the verifiers and the judge model are
imperfect measures; and the sample per category and depth cell is small
(at most 3 questions x 3 runs).

## 3. Two-way tables

Both tables are produced by the tables module described in
`mitra/docs/coherence-tables.md`. Paste the generated output below,
unaltered, and confirm that the row and column definitions match that doc.

### Table 1: mean grammar score by category and depth

```
TBD: paste generated table
```

### Table 2: Hindi-flag rate by category and depth

```
TBD: paste generated table
```

Cells with few records (n) should be read with caution. State the smallest n
here: TBD.

## 4. Scatterplot

![Grammar score vs. Hindi-flag rate](figures/coherence_purity_scatter.png)

TBD: generate the figure with the analysis step and save it as
`mitra/reports/figures/coherence_purity_scatter.png` (or change the path
above to match). Add a one-sentence description of the axes and what one
point represents once the figure exists.

## 5. Regression statistics

These are placed after the graph because they describe the fitted line
shown in section 4. Copy them from the regression output
(`mitra/docs/coherence-regression.md`).

| Statistic | Value |
| --- | --- |
| Model | TBD |
| n (points used) | TBD |
| Slope | TBD |
| Intercept | TBD |
| R^2 | TBD |
| p-value (slope) | TBD |
| Standard error / confidence interval | TBD |

Interpretation: TBD. Say whether the slope is significant, how large the
effect is, and remind readers that this is correlational and that the
sample is small.

## 6. Raw data

| Item | Location |
| --- | --- |
| Harness records (JSONL) | `mitra/logs/harness_records.jsonl`, or the path in `MITRA_HARNESS_LOG` |
| Spoken-line log (JSONL) | `mitra/logs/spoken_lines.jsonl`, or the path in `MITRA_SPOKEN_LOG` |
| Conversation-starter selection | `mitra/data/conversation_starters_v1.json` (version may differ) |
| CSV export | produced by `python -m eval.harness csv <out.csv>` |

The `mitra/logs/` files are not committed to git. TBD: state where the
archived copy of the raw data for this report is kept (for example a
release asset or shared drive) and give its checksum, so the report can be
tied to exact data.

## 7. Reproducing the results

Run all commands from the `mitra/` directory with the virtual environment
active and the environment variables loaded, as in
`mitra/docs/coherence-check-setup.md`.

```bash
# 1. Check judge-model configuration
python -m coherence.check_env

# 2. Fetch and sample the conversation starters (fixed seed 13)
python -m eval.starters fetch
python -m eval.starters show

# 3. Run the campaign against the agent in the simulator
#    (see mitra/docs/coherence-campaign.md for the exact invocation)

# 4. Recompute the per-category, per-depth summary from raw records
python -m eval.harness summary
python -m eval.harness csv results.csv

# 5. Regenerate tables, scatterplot and regression
#    (see mitra/docs/coherence-tables.md, coherence-analysis.md and
#     coherence-regression.md; use `python -m <module> --help` to
#     confirm the flags)
```

Steps 3 and 5 refer to the per-stage docs rather than repeating commands
here, so that this report cannot drift out of sync with the code. When
finalising, replace them with the exact commands that were run.

Unit tests for the analysis code are in `mitra/tests/`:

```bash
python -m unittest discover -s tests
```

Because the judge model is an external service, judged metrics may differ
slightly between runs even with the same data.

## 8. Provenance

- Issue: #13 (agent-MITRA-CoherenceCheck-1)
- Branch: `agent-mitra-coherence-check-1`
- Commit reference for the final commit message: `Add coherence/purity final report (refs #13)`
- Report date and commit hash of the code used: TBD
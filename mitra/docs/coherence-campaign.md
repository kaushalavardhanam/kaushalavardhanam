# Coherence Check: Full Test Campaign (Issue #13, sub-task 8)

`mitra/eval/campaign.py` runs the whole campaign and logs every line.

**Status:** the runner was written without access to the simulator or the
GPT5.6 API, so no campaign has been run and no results are included here.
The steps below must be run locally. The runner's agent driver and the
plug-in function names are assumptions to confirm (see "Assumptions").

## What it does

For each of the 27 categories in the saved starter selection
(`python -m eval.starters fetch`), up to 3 random questions, 3 runs each:

1. **Reachy Mini:** launches `main.py` per dialogue. The seed question
   gets the first agent reply (depth 0). GPT5.6 then plays the human and
   writes Sanskrit replies to the agent's last line; these are fed back to
   `main.py` until `--max-turns` agent turns (default 4).
2. **GPT5.6 baseline:** the same seed question goes to GPT5.6 alone, with
   the same GPT5.6 simulated human generating follow-ups.
3. Every reply line is stored via `eval.harness` with its depth (first
   spoken line = 0), plus `system` (`reachy` / `gpt`), `prompt` and `turn`.
   If `main.py` speaks several lines in one turn, each gets its own depth.

Logs: `logs/campaign_reachy.jsonl`, `logs/campaign_gpt.jsonl`,
`logs/campaign_spoken.jsonl` (raw agent spoken log). Do not commit them.

## Running

From `mitra/`, with the simulator up and `.env` loaded
(`python -m coherence.check_env` must pass):

```bash
python -m eval.campaign --dry-run     # check the plan: 27 categories, dialogue count
python -m eval.campaign \
    --grammar-fn eval.metrics:<grammar function> \
    --hindi-fn   eval.metrics:<hindi function> \
    --scorer     eval.scoring:<function taking (reachy_records, gpt_records)>
```

Other options: `--skip-baseline`, `--skip-reachy`, `--max-turns`, `--runs`,
`--timeout`, `--settle`, `--starters`.

Progress is stored in `logs/campaign_state.jsonl`; re-run the same command
to resume. If a run crashed mid-dialogue, its partial records remain in the
log; remove them (or restart with fresh log files) before scoring.

## Scores

The runner does not define score formulas. Component scores and the
coherence and purity performances come from the earlier modules
(`eval.metrics`, `eval.judged_metrics`, `eval.scoring`). Pass a function
via `--scorer` that takes the Reachy and GPT5.6 record lists and returns a
JSON-serialisable dict; it is written to `logs/campaign_results.json`. If
`--scorer` is omitted, only the per-category, per-depth grammar / Hindi
summary is printed, and the score computation can be done later from the
JSONL files with `eval.harness.load_records()`.

## Assumptions to confirm

- `main.py` reads one line of user text per turn from stdin and logs each
  reply through `log_spoken_line`. If not, write an adapter and pass it via
  `--agent-factory module:function` (object with `ask(text) -> list[str]`
  and `close()`).
- The starters JSON layout is read tolerantly (category to question list,
  or a list of category entries with `questions`/`selected`).
- GPT5.6 is called through the `openai` SDK using `OPENAI_API_KEY`,
  `MITRA_JUDGE_MODEL` and optional `OPENAI_BASE_URL`.
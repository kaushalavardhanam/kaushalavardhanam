# Evaluation harness — variance, and a gate that survives it

**Branch:** `sanskrit-quality-v2` | **Date:** 2026-08-22 | **Related:** [sanskrit-quality-v2.md](sanskrit-quality-v2.md) (open item 1), [`eval/README.md`](../eval/README.md)

A record of the session that built the 50-question evaluation set out into something
CI can rely on. The previous work record closed by naming a 50-question set as "the
single most valuable next thing", on the grounds that ten turns cannot distinguish a
real improvement from a good sample. That turned out to be true of fifty turns as
well, and this session is mostly about measuring by how much. Numbers here are
reproducible with `eval/eval_grammar.py` and `scripts/dry_run.py`.

---

## Summary — X → Y → Z

**X (the problem).** The scorer existed and worked, but its regression gate compared
one run against one run, failing on a clean-rate drop of more than five points or on
*any* per-class increase. Neither threshold had been measured against the harness's
own noise. A laptop freeze mid-session left a suggestive artifact: run C showed nine
`person_mismatch` where the run before it showed none, which read as a sharp
regression and prompted the investigation.

**Y (what was found).** Run C was not a regression, and it was not simple sampling
noise either. It began at 13:15:16; the only source edit in the window was written at
13:15:46, thirty seconds after the process had already imported its modules, and
nothing else in the tree changed between runs B and C — they executed byte-identical
code. The nine failures were also not nine independent errors: they fall in two
contiguous streaks, q05–q10 and q21–q23, each traceable to one bad turn at the head.
`config.yaml` already documents the mechanism at `agent.max_history_turns: 4` — *"Qwen
imitates its own recent output more than the few-shot examples, so a bad phrasing
repeats until it ages out of the window."* Three further runs were then driven on
identical code to establish the spread properly.

**Z (what was changed).** `--turns` now repeats, pooling several runs into one result.
The gate's tolerance is derived from the baseline's own measured spread — two standard
deviations, floored at five points — rather than hardcoded, and a class fails only when
its mean clears the worst the baseline ever recorded. `--save-baseline` refuses fewer
than three runs, since a spread cannot be measured from one. `eval/baseline.json` was
written from three current-code runs: mean 63%, sd 3.4%, tolerance 7%.

---

## The variance, measured

Six runs of the same fifty questions. Runs A–C predate the vocabulary fragment-guard
edit; D–F follow it. Run C is over 46 questions because the freeze truncated it, so its
70% is not comparable to the rest and is excluded from the range below.

| run | clean | no_content | person | tense | hindi | collapse |
| --- | --- | --- | --- | --- | --- | --- |
| A (`logs/eval`) | 31/50 62% | 18 | 1 | 2 | 1 | 2 |
| B (`logs/eval-b`) | 34/50 68% | 16 | 0 | 2 | 2 | 2 |
| C (`logs/eval-c`) | 32/46 70%\* | 11 | **9** | 2 | 1 | 1 |
| D (`logs/eval-d`) | 29/50 58% | 21 | 0 | 3 | 1 | 3 |
| E (`logs/eval-e`) | 32/50 64% | 18 | 4 | 1 | 1 | 1 |
| F (`logs/eval-f`) | 33/50 66% | 16 | 1 | 1 | 1 | 2 |

\* truncated run, 46 of 50 questions.

The five complete runs span **58 to 68 percent — a ten point range with nothing
changed between them**. Sampling is `temperature: 0.3` with no seed. The old gate
failed on a drop of more than five points, which is below that floor: it would have
fired on chance alone, and would have failed run D against run B as a regression when
the two differ only in draw.

`person_mismatch` is the mirror image: 0 or 1 in every complete run, and 9 in the one
that streaked. A gate reacting to any per-class increase treats both the noise and the
streak as the same event.

## Why run C streaked

Read in question order, run C's failures are adjacent rather than scattered:

```
q04         अहं पुस्तकं पठामि।        ← correct
q05 PERSON  अहं संगीतं शृणोति।        ← streak begins
q06–q09 PERSON                         ← imitates it
q10 PERSON  अहं दौड़ते।
q11         अहं गणितं पठामि।          ← clears
…
q21–q23 PERSON                         ← second streak
```

With four exchanges held in context, one third-person verb under अहं at q05 stays
visible for the next several turns and is copied; the pattern clears at q11, roughly
window-plus-two turns after it started. This is the documented behaviour of the history
window, not a new fault, and it means **the model does not fail per reply** — it fails
per incident, and the scorer counts replies.

## What was built

- **`aggregate()` / `report_aggregate()`** — pool N runs, reporting mean, standard
  deviation, range, per-run rates, and per-class mean/max/per-run counts.
- **`compare()` reworked** — gates the pooled mean against a tolerance carried in the
  baseline, and gates each error class on `mean > max(baseline_max, baseline_mean + 1)`.
- **`--turns` repeats**; `--save-baseline` exits 2 on fewer than three runs.
- **`eval/baseline.json`** — three runs, mean 63%, sd 3.4%, range 58–66%, tolerance 7%.

The baseline's class profile, which is the part that tells you whether a fix landed:

| class | mean | max | per run |
| --- | --- | --- | --- |
| `no_content` | 18.33 | 21 | 21, 18, 16 |
| *sandhi* (warn) | 4.33 | 7 | 7, 4, 2 |
| `tense_mismatch` | 2.67 | 4 | 3, 4, 1 |
| *collapse* | 2.0 | 3 | 3, 1, 2 |
| *persona_gender* (warn) | 1.67 | 3 | 1, 1, 3 |
| `hindi_marker` | 1.0 | 1 | 1, 1, 1 |
| `forbidden_word` | 1.0 | 1 | 1, 1, 1 |
| `person_mismatch` | 0.33 | 1 | 0, 0, 1 |

## Verification

| input | headline | gate |
| --- | --- | --- |
| D/E/F against their own baseline | 63% → 63% | exit 0, no regression |
| A/B/C against that baseline | 63% → **66%** | exit 1, fails on `person_mismatch` 0.33 → 3.33 |

The second row is the one worth keeping. The pre-edit runs score *higher* on the
headline and still fail, which is what the harness was for: the per-class counts, not
the clean rate, are what say whether a fix landed. The old gate would have passed that
comparison on its headline and failed the first row on noise — exactly backwards.

Unit tests remain 148 passed, 6 skipped, with the vocabulary fragment-guard edit in
place.

## Open items

1. **Count incidents, not replies.** The gate still over-weights a streak: run C's two
   bad turns are scored as nine failures. Collapsing a run of the same error class in
   consecutive questions into one incident would make per-class gating far steadier,
   and is the obvious next change to the scorer.
2. **`no_content` is the dominant class at ~18 per run** and is mostly genuine. Only 1–2
   flags per run come from the scorer missing a म्/ं variant (`नृत्यम्` against a listed
   `नृत्यं`, `आम्रफलं` against `आम्रम्`); widening `content_any` buys about 4%, not the
   class. The rest are real — `क्षम्यताम्, अहं न अवगच्छामि` fallbacks, and near-misses
   like `अहं चलामि` for "do you run".
3. **The baseline pins current-tree code**, including the uncommitted `vocabulary.py` /
   `grammar.py` fragment-guard edit. Reverting or amending that edit invalidates it;
   regenerating costs three runs, about four minutes.
4. **No seed.** `temperature: 0.3` with no seed is why three runs are needed for a
   baseline at all. A seed would not make the eval honest — the child hears sampled
   output — but it would make bisecting a real regression much cheaper.
5. **Two files named `eval_grammar.py`.** `eval/eval_grammar.py` scores a run against
   the 50 questions; `scripts/eval_grammar.py` measures the validator's false-positive
   rate against the 981-sentence phrasebook. Different inputs, different questions,
   confusable names.

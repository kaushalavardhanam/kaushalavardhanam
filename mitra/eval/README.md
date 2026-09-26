# Mitra grammar evaluation

Fifty questions and a deterministic scorer, so that a number like "8 of 9 clean"
can be checked rather than believed. Runs A to D differed by 10, 15, 45 and 89
percent on identical code, which is the whole reason this exists: at ten turns,
a real improvement and a lucky sample look the same.

## Files

- `questions.yaml` — the fifty prompts with the expectations each answer must meet
- `eval_grammar.py` — the scorer, pure string and morphology matching, no model calls
- `baseline.json` — written by `--save-baseline`, compared against by `--baseline`

## Running it

```bash
# 1. print the prompts in order, to read aloud or feed to the pipeline
python3 eval/eval_grammar.py --list

# 2. drive a run, then score the turn log it produced
python3 scripts/dry_run.py --questions eval/questions.yaml --log-dir logs/eval-d
python3 eval/eval_grammar.py --turns logs/eval-d/turns.jsonl

# 3. freeze the current state over SEVERAL runs, then check later ones against it
python3 eval/eval_grammar.py \
    --turns logs/eval-d/turns.jsonl \
    --turns logs/eval-e/turns.jsonl \
    --turns logs/eval-f/turns.jsonl \
    --save-baseline eval/baseline.json
python3 eval/eval_grammar.py --turns logs/eval-g/turns.jsonl ... --baseline eval/baseline.json
```

`--turns` repeats, and pooling is the point. One run is not a measurement: five
complete runs of near-identical code scored 58, 62, 64, 66 and 68 percent, a ten
point range with nothing changed between them. A gate that failed on a five
point drop, as the first version did, fires on chance alone.

So the gate derives its own tolerance from the baseline's spread — two standard
deviations, floored at five points — and `--save-baseline` refuses fewer than
three runs, since a spread cannot be measured from one. Exit is non-zero when
the pooled mean falls further than that tolerance, or when an error class's mean
clears the worst the baseline ever saw. It drops into CI next to the 148 unit
tests.

The class check is the half that earns its keep. Scoring the three pre-edit runs
against a three-run baseline gives a clean rate of 66 percent against 63, an
apparent *improvement*, while `person_mismatch` goes from a mean of 0.33 to
3.33 and correctly fails the gate.

Turns are matched to questions by `question_id` when the log carries one, and
otherwise by fuzzy match on the transcript, which tolerates the mis-hearings the
logs already show ("Two play sports", "Will it be my friend"). Writing
`question_id` into each turn record removes the guesswork and is worth the one
line, which `scripts/dry_run.py --questions` already writes.

### One caveat on the class counts

A failure is counted per reply, but the model does not fail per reply. With
`agent.max_history_turns: 4` it imitates its own recent output, so one bad
phrasing repeats until it ages out of the window: run C's nine
`person_mismatch` were not nine independent errors but two streaks, q05-q10 and
q21-q23, each traceable to a single bad turn at the head. The gate therefore
still over-weights a streak. Counting incidents rather than replies would fix
it and is the obvious next change.

## What counts as clean

A reply is clean when it trips none of these:

| class | what it catches | seen in |
| --- | --- | --- |
| `hindi_marker` | खेलानि, घरे, आज, मक्खनम्, दूधम्, दालः | A, B, C |
| `nonword` | करोष्यसि, कुरुमि, पठमि, खेलनी, अभवति | A, B, C |
| `person_mismatch` | अहं with a third person verb, भवान् with a second person verb, त्वम् with अस्मि | all |
| `frame_broken` | अहं X प्रियं अस्मि, where मम X अस्ति or मह्यं X रोचते is wanted | A, B, C |
| `no_content` | grammatical but answers nothing, the "my favourite food exists" case | D |
| `forbidden_word` | question-specific, e.g. दालः on the food question | C |
| `tense_mismatch` | future question answered in the present, and the reverse | A, B, C |
| `gerundive` | अध्ययनीयम्, स्मरणीयम्, ज्ञातव्यम् used as a finite verb | B, C |

Two more are reported but do not fail a reply, since neither is the validator's
job today:

- `sandhi` — anusvara before a vowel, `भोजनं अस्ति` for `भोजनम् अस्ति`
- `persona_gender` — masculine self-description such as `अवस्थितः`, against the
  neuter `मित्रम्` the prompt declares

Two further figures are counted rather than judged:

- **template** — how many replies ask a question back. With
  `session.max_sentences: 1` this should be zero. If it climbs, the truncation
  that removed the template lock in run D has stopped applying, which matters
  because that fix was truncation and not grammar.
- **collapse** — near-synonym questions returning byte-identical replies. Four
  pairs are seeded for this: play/sports, live/house, today twice, and who/about
  yourself. The play and sports pair collapsed in four consecutive runs.

`no_content` is the check worth watching after a retry-instruction change. The
first version of that instruction told the model to remove the offending word,
and it complied by deleting the answer. A reply that passes every morphology
check and says nothing is exactly what this class is for.

## Reading the result

The headline is the clean rate over answered questions. Beneath it, the per-class
counts are what tell you whether a fix landed: those classes were 100 percent
reproducible across the first thirty turns, so any of them returning to non-zero
is a regression regardless of what the headline does.

Categories are roughly ten questions of daily actions, eight of preferences, six
each of place, tense, polarity and identity, and six of simple facts. A clean
rate that is high overall but zero in one category is more useful than either
number alone.

## Extending it

Add a question by appending to `questions.yaml`. `content_any` is the field that
does the real work: list the words that would constitute an actual answer. Leave
it out and the question only gets the global checks, which is fine for prompts
where any sensible reply is acceptable.

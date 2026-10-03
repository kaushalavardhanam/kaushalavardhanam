# Coherence Check: Conversation Starters (Issue #13, sub-task 3)

`mitra/eval/starters.py` parses the daily-dictionary page
<https://sanskritdocuments.org/doc_z_misc_major_works/daily.html> into
categories (27 expected), then randomly selects up to 3 questions per
category with a fixed, recorded seed. Each selected question is tested 3
times with the model.

## Run

From the `mitra/` directory:

```bash
python -m eval.starters fetch          # download, parse, sample, save
python -m eval.starters show           # print the saved selection
```

Output: `mitra/data/conversation_starters_v1.json` (commit this file).

- Seed: `13` (stored in the file as `seed`, along with `source_sha256` and
  `fetched_at`). Each category uses its own RNG seeded from `seed:index`, so
  the selection is reproducible for the same page content.
- The file is versioned by name. An existing file is never overwritten unless
  `--force` is given; use `--version v2` for a new selection.
- `runs_per_question` is stored as `3`.

## Parsing assumptions (please verify)

This was written without inspecting the live page, so the parser is
heuristic:

- A category starts at each heading tag (default `h2,h3`, change with
  `--heading-tags`).
- A candidate question is any text line in a category containing `?`.
- If the category count is not 27, the command prints the count and exits
  without saving. Adjust `--heading-tags` / `--split-newlines`, or use
  `--allow-mismatch` after checking the result manually.
- Categories with fewer than 3 questions are reported and kept as-is.
- `--html saved.html` parses an offline copy.

## Using the selection

```python
from eval.starters import load_selection, iter_trials
from eval.harness import Harness

sel = load_selection("v1")
harness = Harness(...)
for category, question, run in iter_trials(sel):   # run is 1..3
    conv = harness.conversation(category, question, run)
    ...
```

## Checklist

- [ ] `fetch` reports 27 categories
- [ ] Selection reviewed by hand (questions look like real questions)
- [ ] `conversation_starters_v1.json` committed
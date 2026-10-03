# Coherence Check: Score Normalization and Performance Formulas (Issue #13, sub-task 7)

`mitra/eval/scoring.py` defines how single-component values are normalized and
how they are combined into test results.

## Terminology

- **Score**: the value of a *single component*. A raw score is what the metric
  produces; a *normalized score* is that value mapped to 0–1 with **higher
  always better**.
- **Performance**: the *combined* test result built from several normalized
  scores. There are two: coherence performance and purity performance.

Do not call a combined value a "score", and do not call a single component
value a "performance".

## Normalization

Each component is declared with a `ComponentSpec`:

| Field | Meaning |
| --- | --- |
| `name` | Component name (unique) |
| `direction` | `"+"` higher raw value is better, `"-"` lower raw value is better |
| `min_value`, `max_value` | Raw range (default 0 and 1, e.g. rates and 0–1 scores) |
| `in_purity` | `True` if the component is one of the 7 purity components |

Let `f = (clamp(raw, min, max) - min) / (max - min)`. Then:

- direction `+`: `normalized = f`
- direction `-`: `normalized = 1 - f`

So for a `[-]` component such as a Hindi rate, a raw 0.2 becomes 0.8. Raw
values outside the range are clamped. Missing or NaN values raise
`ValueError` rather than being silently treated as 0.

When a component has several runs or depths, average the raw values first
(`mean_score`, which ignores `None`), then normalize the average.

## Formulas

```
coherence performance = mean(normalized score of all 10 components)
purity performance    = mean(normalized score of the 7 purity components)
```

Both are equal-weight arithmetic means and lie in [0, 1]; higher is better.

## Usage

Component names are not hard-coded in the module. Build the scheme from the
component list in the issue; `build_scheme` enforces exactly 10 components
and exactly 7 with `in_purity=True`:

```python
from eval.scoring import ComponentSpec, build_scheme

scheme = build_scheme([
    ComponentSpec("component_a", "+", in_purity=True),
    ComponentSpec("component_b", "-"),
    # ... 10 in total, 7 with in_purity=True
])
scheme.coherence_performance(raw_scores)   # raw_scores: {name: raw value}
scheme.purity_performance(raw_scores)
```

Set the `direction` of every `[-]` component in the issue to `"-"`, and check
the raw range of each metric in `eval/metrics.py` and
`eval/judged_metrics.py` when declaring the specs.

## Tests

```bash
python -m unittest tests.test_scoring     # run from mitra/
```
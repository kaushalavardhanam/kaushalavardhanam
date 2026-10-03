# Coherence Check: Scatterplot and Regression (Issue #13, sub-task 10)

`mitra/eval/regression.py` plots dialogue depth (x) against coherence (red
dots) and purity (blue dots), fits a least-squares line `y = a + bx` to each
series, and saves the plot and statistics as data files.

## Input

A CSV or JSONL file with one row per observation and the fields `depth`,
`coherence` and `purity` (override names with `--depth-col`,
`--coherence-col`, `--purity-col`). Rows with a missing or non-numeric value
are skipped for that series only. Use `--aggregate` to average each series per
depth before fitting (fits then use one point per depth).

## Run (from `mitra/`)

```bash
python -m eval.regression scores.csv
python -m eval.regression scores.jsonl --aggregate --prefix run1
```

Requires `matplotlib`; the statistics are computed in pure Python.

## Outputs (default `mitra/data/`)

| File | Contents |
| --- | --- |
| `<prefix>_scatter.png` | Red = coherence, blue = purity, with regression lines, equations and r² |
| `<prefix>_stats.json` / `.csv` | Per-series statistics |
| `<prefix>_residuals.csv` | `series, depth, observed, fitted, residual` |
| `<prefix>_lines.csv` | Regression line value at each observed depth |

Default prefix: `depth_regression`.

## Statistics per series

`n`, `intercept` (a), `slope` (b), `equation`, `r`, `r2`, `adjusted_r2`,
`slope_stderr`, `intercept_stderr`, `t_statistic`, `p_value` (two-sided test of
slope = 0), `residual_stderr`, `sse`, `sst`, `x_mean`, `y_mean`, `note`.

Values that cannot be computed are `null`: fewer than 2 points gives no fit;
2 points give an exact line without standard errors or p-value; constant
y gives no r or r²; identical depths give no slope.

The script was written without being run against real campaign data;
check the outputs on a small file first.
</parameter>
# Coherence Check: Two-Way Result Tables (Issue #13, sub-task 9)

`mitra/eval/tables.py` aggregates result records into two markdown tables,
one for **coherence** and one for **purity**.

## Layout

- Rows: dialogue depth (first spoken line = 0).
- Columns: component (question category / component under test).
- Each cell: `mean ± sd` over the individual records in that cell.
- Subtotal column (`All components`): performance per depth.
- Subtotal row (`All depths`): overall score per component regardless of depth.
- Bottom-right cell: grand total.

Subtotals pool the underlying records; they are not averages of cell means.
The standard deviation is the sample standard deviation (n - 1) and is shown
as `-` for cells with fewer than two values. Records with a missing metric
value are skipped for that metric.

## Input

JSONL, one record per line, with `depth`, `component` (or `category`),
`coherence` and `purity`. Key names can be changed with `--depth-key`,
`--component-key`, `--coherence-key`, `--purity-key`. If the campaign output
uses different field names, pass them with these options.

## GPT5.6 comparison

Pass `--compare` with a JSONL file in the same format, produced by running
the same questions against GPT5.6. It is rendered as a second table per
metric, aligned to the same depths and components as the MITRA table.

## Usage

Run from the `mitra/` directory:

```bash
python -m eval.tables results.jsonl
python -m eval.tables results.jsonl --compare gpt56_results.jsonl --out logs/tables.md
```

From Python: `eval.tables.build_report(records, comparison=gpt_records)`.

Note: this was written without access to the exact record layout produced by
the earlier campaign step, so the field names above are configurable
assumptions. Check them against real output before relying on the tables.
</gr_replace>
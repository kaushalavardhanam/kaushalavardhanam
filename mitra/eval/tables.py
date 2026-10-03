"""Aggregate coherence and purity results into two-way markdown tables.

Two tables are produced (one for coherence, one for purity):

    rows     dialogue depth (first spoken line = 0)
    columns  component (question category / component under test)

Every cell shows "mean ± sd" over the individual result records that fall in
that (depth, component) cell. A subtotal column ("All components") gives the
performance per depth, and a subtotal row ("All depths") gives the overall
score per component regardless of depth. The bottom-right corner is the grand
total. Subtotals are computed by pooling the underlying records, not by
averaging the cell means. The standard deviation is the sample standard
deviation (n - 1); it is shown as "-" when a cell has fewer than two values.

A second set of tables with the GPT5.6 comparison values is added when a
comparison file is supplied. It uses the same record format, produced by
running the same questions against GPT5.6, and is aligned to the same rows
and columns.

Input records are JSONL, one object per line, with at least:
    depth       int
    component   str   (falls back to "category" if "component" is absent)
    coherence   float (records where it is missing/null are skipped)
    purity      float (records where it is missing/null are skipped)
The key names can be changed with the command-line options.

Usage (from the mitra/ directory):

    python -m eval.tables results.jsonl
    python -m eval.tables results.jsonl --compare gpt56_results.jsonl --out tables.md
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROW_TOTAL_LABEL = "All components"
COL_TOTAL_LABEL = "All depths"


def load_jsonl(path):
    """Return the records of a JSONL file as a list of dicts."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _component(record, component_key):
    value = record.get(component_key)
    if value is None and component_key == "component":
        value = record.get("category")
    return value


def collect(records, value_key, component_key="component", depth_key="depth"):
    """Return {(depth, component): [values]} for records with a numeric value."""
    cells = {}
    for r in records:
        value = _to_float(r.get(value_key))
        depth = r.get(depth_key)
        component = _component(r, component_key)
        if value is None or depth is None or component is None:
            continue
        cells.setdefault((depth, str(component)), []).append(value)
    return cells


def stats(values):
    """Return (mean, sd, n); mean/sd are None when not defined."""
    n = len(values)
    if n == 0:
        return (None, None, 0)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n >= 2 else None
    return (mean, sd, n)


def format_cell(values, digits=3):
    """Format a list of values as 'mean ± sd'."""
    mean, sd, n = stats(values)
    if mean is None:
        return "-"
    sd_text = "-" if sd is None else f"{sd:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd_text}"


def _esc(text):
    return str(text).replace("|", "\\|")


def _sorted_depths(depths):
    return sorted(depths, key=lambda d: (not isinstance(d, (int, float)), d if isinstance(d, (int, float)) else str(d)))


def render_table(cells, depths, components, digits=3):
    """Render a two-way markdown table with subtotal row and column.

    cells: {(depth, component): [values]}; depths/components define the layout.
    """
    header = ["Depth"] + [_esc(c) for c in components] + [ROW_TOTAL_LABEL]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(components) + 1)) + " |",
    ]
    for depth in depths:
        row_values = []
        row = [str(depth)]
        for comp in components:
            vals = cells.get((depth, comp), [])
            row.append(format_cell(vals, digits))
            row_values.extend(vals)
        row.append(format_cell(row_values, digits))
        lines.append("| " + " | ".join(row) + " |")

    total_row = [f"**{COL_TOTAL_LABEL}**"]
    grand = []
    for comp in components:
        vals = [v for d in depths for v in cells.get((d, comp), [])]
        total_row.append(format_cell(vals, digits))
        grand.extend(vals)
    total_row.append(format_cell(grand, digits))
    lines.append("| " + " | ".join(total_row) + " |")
    return "\n".join(lines)


def build_report(
    records,
    comparison=None,
    metrics=(("Coherence", "coherence"), ("Purity", "purity")),
    component_key="component",
    depth_key="depth",
    digits=3,
    model_label="MITRA",
    comparison_label="GPT5.6",
):
    """Build the full markdown report (coherence and purity tables)."""
    sections = ["# Coherence and purity by dialogue depth and component", ""]
    sections.append(
        "Cells show mean ± standard deviation. The last column is the "
        "subtotal per depth; the last row is the overall score per component "
        "regardless of depth."
    )
    sections.append("")
    for title, key in metrics:
        main = collect(records, key, component_key, depth_key)
        comp = collect(comparison, key, component_key, depth_key) if comparison else {}
        depths = _sorted_depths({d for d, _ in main} | {d for d, _ in comp})
        components = sorted({c for _, c in main} | {c for _, c in comp})

        sections.append(f"## {title} - {model_label}")
        sections.append("")
        if not main:
            sections.append(f"_No records with a `{key}` value._")
        else:
            sections.append(render_table(main, depths, components, digits))
        sections.append("")
        if comparison is not None:
            sections.append(f"## {title} - {comparison_label} comparison")
            sections.append("")
            if not comp:
                sections.append(f"_No {comparison_label} records with a `{key}` value._")
            else:
                sections.append(render_table(comp, depths, components, digits))
            sections.append("")
    return "\n".join(sections).rstrip() + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Two-way coherence/purity tables")
    parser.add_argument("results", help="JSONL results for the MITRA agent")
    parser.add_argument("--compare", help="JSONL results for GPT5.6 (same format)")
    parser.add_argument("--out", help="write the markdown to this file")
    parser.add_argument("--component-key", default="component")
    parser.add_argument("--depth-key", default="depth")
    parser.add_argument("--coherence-key", default="coherence")
    parser.add_argument("--purity-key", default="purity")
    parser.add_argument("--digits", type=int, default=3)
    args = parser.parse_args(argv)

    records = load_jsonl(args.results)
    comparison = load_jsonl(args.compare) if args.compare else None
    report = build_report(
        records,
        comparison=comparison,
        metrics=(("Coherence", args.coherence_key), ("Purity", args.purity_key)),
        component_key=args.component_key,
        depth_key=args.depth_key,
        digits=args.digits,
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"Wrote tables to {out}")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
</gr_replace>
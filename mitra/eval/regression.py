"""Scatterplot and regression statistics: dialogue depth vs coherence and purity.

Reads per-observation (or per-depth) scores from a CSV or JSONL file with the
columns/fields ``depth``, ``coherence`` and ``purity`` (names configurable),
fits an ordinary least-squares line y = a + b*x to each series, plots both
series and writes the results as data files.

Usage (from the mitra/ directory):

    python -m eval.regression scores.csv
    python -m eval.regression scores.jsonl --aggregate      # mean per depth first
    python -m eval.regression scores.csv --outdir data --prefix run1

Outputs (default directory mitra/data/, default prefix "depth_regression"):

    <prefix>_scatter.png       red dots = coherence, blue dots = purity, with fitted lines
    <prefix>_stats.json        per-series statistics (a, b, r, r2, SEs, p-value, n, ...)
    <prefix>_stats.csv         the same, one row per series
    <prefix>_residuals.csv     series, depth, observed, fitted, residual
    <prefix>_lines.csv         series, depth, fitted (regression line at each x)

Statistics per series (n >= 2 needed; some need n >= 3):
    n, intercept (a), slope (b), equation, r (correlation), r2, adjusted_r2,
    slope_stderr, intercept_stderr, t_statistic, p_value (two-sided test of
    slope == 0, equal to the correlation test), residual_stderr, sse, sst,
    x_mean, y_mean.

Only matplotlib is needed (for the image); the statistics are pure Python.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_PREFIX = "depth_regression"
SERIES = (
    ("coherence", "red"),
    ("purity", "blue"),
)


# ---------------------------------------------------------------- statistics

def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t, df):
    """Two-sided p-value for a Student t statistic with df degrees of freedom."""
    if df <= 0:
        return None
    if math.isinf(t):
        return 0.0
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def _fmt_num(v):
    return f"{v:.6g}"


def linear_fit(xs, ys):
    """Ordinary least squares fit of y = a + b*x.

    Returns a dict of statistics plus "fitted" and "residuals" lists.
    Values that cannot be computed are None.
    """
    n = len(xs)
    result = {
        "n": n,
        "intercept": None,
        "slope": None,
        "equation": None,
        "r": None,
        "r2": None,
        "adjusted_r2": None,
        "slope_stderr": None,
        "intercept_stderr": None,
        "t_statistic": None,
        "p_value": None,
        "residual_stderr": None,
        "sse": None,
        "sst": None,
        "x_mean": None,
        "y_mean": None,
        "note": None,
        "fitted": [],
        "residuals": [],
    }
    if n < 2:
        result["note"] = "need at least 2 points"
        return result
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    result["x_mean"], result["y_mean"] = xbar, ybar
    sxx = sum((x - xbar) ** 2 for x in xs)
    syy = sum((y - ybar) ** 2 for y in ys)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    if sxx == 0:
        result["note"] = "all depths identical; slope undefined"
        return result

    b = sxy / sxx
    a = ybar - b * xbar
    fitted = [a + b * x for x in xs]
    resid = [y - f for y, f in zip(ys, fitted)]
    sse = sum(e * e for e in resid)

    result.update(
        intercept=a,
        slope=b,
        equation=f"y = {_fmt_num(a)} + {_fmt_num(b)}x",
        sse=sse,
        sst=syy,
        fitted=fitted,
        residuals=resid,
    )
    if syy > 0:
        result["r"] = max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))
        result["r2"] = max(0.0, 1.0 - sse / syy)
        if n > 2:
            result["adjusted_r2"] = 1.0 - (1.0 - result["r2"]) * (n - 1) / (n - 2)
    else:
        result["note"] = "y is constant; r and r2 undefined"

    if n > 2:
        df = n - 2
        s2 = sse / df
        result["residual_stderr"] = math.sqrt(s2)
        se_b = math.sqrt(s2 / sxx)
        se_a = math.sqrt(s2 * (1.0 / n + xbar * xbar / sxx))
        result["slope_stderr"] = se_b
        result["intercept_stderr"] = se_a
        if se_b > 0:
            t = b / se_b
            result["t_statistic"] = t
            result["p_value"] = t_two_sided_p(t, df)
        else:  # perfect fit
            result["p_value"] = 0.0 if b != 0 else 1.0
    else:
        extra = "only 2 points: line is exact, no standard errors or p-value"
        result["note"] = f"{result['note']}; {extra}" if result["note"] else extra
    return result


# ------------------------------------------------------------------ data I/O

def _to_float(value):
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def load_rows(path):
    """Load rows (list of dicts) from a .csv or .jsonl/.json file."""
    p = Path(path)
    if p.suffix.lower() == ".csv":
        with p.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    with p.open("r", encoding="utf-8") as f:
        text = f.read()
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def extract_series(rows, depth_col, value_col, aggregate=False):
    """Return (xs, ys) for one series; optionally the mean y per depth."""
    pairs = []
    for row in rows:
        x = _to_float(row.get(depth_col))
        y = _to_float(row.get(value_col))
        if x is not None and y is not None:
            pairs.append((x, y))
    if aggregate:
        groups = defaultdict(list)
        for x, y in pairs:
            groups[x].append(y)
        pairs = [(x, sum(v) / len(v)) for x, v in sorted(groups.items())]
    else:
        pairs.sort()
    return [p[0] for p in pairs], [p[1] for p in pairs]


# -------------------------------------------------------------------- output

STAT_FIELDS = [
    "series", "n", "intercept", "slope", "equation", "r", "r2", "adjusted_r2",
    "slope_stderr", "intercept_stderr", "t_statistic", "p_value",
    "residual_stderr", "sse", "sst", "x_mean", "y_mean", "note",
]


def write_outputs(data, fits, outdir, prefix, title=None):
    """Write the PNG, JSON and CSV files. data[name] = (xs, ys). Returns paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "plot": outdir / f"{prefix}_scatter.png",
        "stats_json": outdir / f"{prefix}_stats.json",
        "stats_csv": outdir / f"{prefix}_stats.csv",
        "residuals": outdir / f"{prefix}_residuals.csv",
        "lines": outdir / f"{prefix}_lines.csv",
    }

    stats = {
        name: {k: v for k, v in fit.items() if k not in ("fitted", "residuals")}
        for name, fit in fits.items()
    }
    with paths["stats_json"].open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        f.write("\n")

    with paths["stats_csv"].open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STAT_FIELDS)
        writer.writeheader()
        for name, s in stats.items():
            writer.writerow({"series": name, **{k: s.get(k) for k in STAT_FIELDS[1:]}})

    with paths["residuals"].open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["series", "depth", "observed", "fitted", "residual"])
        for name, fit in fits.items():
            xs, ys = data[name]
            for x, y, fv, r in zip(xs, ys, fit["fitted"], fit["residuals"]):
                writer.writerow([name, x, y, fv, r])

    with paths["lines"].open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["series", "depth", "fitted"])
        for name, fit in fits.items():
            xs, _ = data[name]
            for x, fv in zip(xs, fit["fitted"]):
                writer.writerow([name, x, fv])

    _plot(data, fits, paths["plot"], title)
    return paths


def _plot(data, fits, out_path, title=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color in SERIES:
        if name not in data:
            continue
        xs, ys = data[name]
        fit = fits[name]
        label = name
        if fit["slope"] is not None:
            label = f"{name}: {fit['equation']}"
            if fit["r2"] is not None:
                label += f", r²={fit['r2']:.3f}"
        if xs:
            ax.scatter(xs, ys, color=color, alpha=0.7, s=25, label=label)
        if fit["slope"] is not None:
            x0, x1 = min(xs), max(xs)
            ax.plot(
                [x0, x1],
                [fit["intercept"] + fit["slope"] * x0, fit["intercept"] + fit["slope"] * x1],
                color=color,
                linewidth=1.5,
            )
    ax.set_xlabel("Dialogue depth (first spoken line = 0)")
    ax.set_ylabel("Score")
    ax.set_title(title or "Coherence and purity vs dialogue depth")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------- CLI

def run(input_path, outdir=DATA_DIR, prefix=DEFAULT_PREFIX, depth_col="depth",
        coherence_col="coherence", purity_col="purity", aggregate=False, title=None):
    rows = load_rows(input_path)
    columns = {"coherence": coherence_col, "purity": purity_col}
    data, fits = {}, {}
    for name, _color in SERIES:
        xs, ys = extract_series(rows, depth_col, columns[name], aggregate)
        data[name] = (xs, ys)
        fits[name] = linear_fit(xs, ys)
    paths = write_outputs(data, fits, outdir, prefix, title)
    return fits, paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Depth vs coherence/purity regression")
    parser.add_argument("input", help="CSV or JSONL with depth, coherence, purity")
    parser.add_argument("--outdir", default=str(DATA_DIR), help="output directory")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="output file prefix")
    parser.add_argument("--depth-col", default="depth")
    parser.add_argument("--coherence-col", default="coherence")
    parser.add_argument("--purity-col", default="purity")
    parser.add_argument("--aggregate", action="store_true",
                        help="average each series per depth before fitting")
    parser.add_argument("--title", help="plot title")
    args = parser.parse_args(argv)

    try:
        fits, paths = run(
            args.input, args.outdir, args.prefix, args.depth_col,
            args.coherence_col, args.purity_col, args.aggregate, args.title,
        )
    except FileNotFoundError:
        print("Input not found:", args.input, file=sys.stderr)
        return 1

    for name, fit in fits.items():
        r2 = "-" if fit["r2"] is None else f"{fit['r2']:.4f}"
        p = "-" if fit["p_value"] is None else f"{fit['p_value']:.4g}"
        print(f"{name}: n={fit['n']} {fit['equation'] or 'no fit'} r2={r2} p={p}")
        if fit["note"]:
            print(f"  note: {fit['note']}")
    for key, path in paths.items():
        print(f"Wrote {key}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
</parameter>
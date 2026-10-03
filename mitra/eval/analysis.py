"""Analyze harness records and derive improvement suggestions.

Reads the records written by ``eval.harness`` and reports:

  * which components (categories) drag grammar score and purity down,
  * issue types ranked by ease of fix (empty replies and Latin leakage before
    deep grammar problems),
  * depth trends (does quality drop as the dialogue continues?),
  * the gap between the local grammar verifier and the GPT5.6 judge, when
    records carry a ``judge_score`` field (an extra field passed to
    ``Conversation.log_line(text, judge_score=...)``; assumed to be on the
    same 0..1 scale as ``grammar_score``),
  * concrete recommendations generated from the observed issue rates.

Issue detection is purely heuristic and works on the logged text and flags:

  empty        reply is empty or whitespace only
  latin        Latin letters exceed LATIN_THRESHOLD of all Latin+Devanagari letters
  repetition   long reply with few distinct tokens (looping output)
  hindi        the Hindi detector flagged the reply (hindi_flag true)
  low_grammar  grammar_score below GRAMMAR_THRESHOLD

Usage (from the mitra/ directory):

    python -m eval.analysis report                # markdown report to stdout
    python -m eval.analysis report --path X.jsonl
    python -m eval.analysis report --json         # machine-readable output

The report only describes the data it is given; it makes no claims when
there are no records.
"""

import argparse
import json
import re
import sys
from collections import defaultdict

from eval.harness import load_records

GRAMMAR_THRESHOLD = 0.5
LATIN_THRESHOLD = 0.05
REPETITION_MIN_TOKENS = 6
REPETITION_UNIQUE_RATIO = 0.5
DEPTH_SLOPE_THRESHOLD = 0.01

_LATIN = re.compile(r"[A-Za-z]")
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")

# Ease of fix: 1 = easiest. Each entry: (ease, description, fix hint).
ISSUES = {
    "empty": (
        1,
        "empty reply",
        "Verifier-in-the-loop retry: if the reply is blank, resample (up to N tries) "
        "and fall back to a fixed Sanskrit acknowledgement instead of staying silent.",
    ),
    "latin": (
        2,
        "Latin-script leakage",
        "Post-filter and decoding constraint: ban or penalize Latin-script tokens "
        "(logit bias / bad-words list), and retry or transliterate when the filter fires.",
    ),
    "repetition": (
        3,
        "repetitive / looping output",
        "Decoding change: add a repetition penalty or no-repeat-ngram size, lower "
        "max length, and cut off at the first repeated sentence.",
    ),
    "hindi": (
        4,
        "Hindi instead of Sanskrit",
        "Prompt change: state 'reply only in Sanskrit, never Hindi' with 1-2 short "
        "Sanskrit example turns; keep the Hindi detector as a retry trigger.",
    ),
    "low_grammar": (
        5,
        "low grammar score (deep grammar issues)",
        "Hardest: use the grammar verifier as a reranker over several samples, "
        "shorten replies (fewer clauses, fewer errors), and only then consider "
        "fine-tuning or few-shot examples targeted at the weakest categories.",
    ),
}


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def latin_fraction(text):
    """Share of Latin letters among Latin + Devanagari letters (0 if none)."""
    latin = len(_LATIN.findall(text or ""))
    deva = len(_DEVANAGARI.findall(text or ""))
    total = latin + deva
    return latin / total if total else 0.0


def is_repetitive(text):
    tokens = (text or "").split()
    if len(tokens) < REPETITION_MIN_TOKENS:
        return False
    return len(set(tokens)) / len(tokens) < REPETITION_UNIQUE_RATIO


def detect_issues(record, grammar_threshold=GRAMMAR_THRESHOLD, latin_threshold=LATIN_THRESHOLD):
    """Return the list of issue names that apply to one record."""
    text = record.get("text") or ""
    issues = []
    if not text.strip():
        issues.append("empty")
    else:
        if latin_fraction(text) > latin_threshold:
            issues.append("latin")
        if is_repetitive(text):
            issues.append("repetition")
    if record.get("hindi_flag"):
        issues.append("hindi")
    score = record.get("grammar_score")
    if score is not None and score < grammar_threshold:
        issues.append("low_grammar")
    return issues


def issue_rates(records):
    """Fraction of records affected by each issue type."""
    n = len(records)
    counts = defaultdict(int)
    for r in records:
        for issue in detect_issues(r):
            counts[issue] += 1
    return {issue: (counts[issue] / n if n else 0.0) for issue in ISSUES}


def rank_issues(records):
    """Issues that occur, easiest fix first; ties broken by higher prevalence."""
    rates = issue_rates(records)
    ranked = [
        {
            "issue": issue,
            "description": ISSUES[issue][1],
            "ease": ISSUES[issue][0],
            "rate": rate,
            "fix": ISSUES[issue][2],
        }
        for issue, rate in rates.items()
        if rate > 0
    ]
    ranked.sort(key=lambda d: (d["ease"], -d["rate"]))
    return ranked


def by_category(records):
    """Per-category stats, worst (largest drag on grammar / most issues) first.

    drag = overall mean grammar - category mean grammar (positive = worse
    than average); None when scores are missing.
    """
    overall = _mean([r.get("grammar_score") for r in records])
    groups = defaultdict(list)
    for r in records:
        groups[r.get("category")].append(r)
    rows = []
    for category, recs in groups.items():
        mean_grammar = _mean([r.get("grammar_score") for r in recs])
        rates = issue_rates(recs)
        rows.append(
            {
                "category": category,
                "n": len(recs),
                "mean_grammar": mean_grammar,
                "drag": (overall - mean_grammar)
                if overall is not None and mean_grammar is not None
                else None,
                "issue_rate": sum(1 for r in recs if detect_issues(r)) / len(recs),
                "rates": rates,
            }
        )
    rows.sort(key=lambda d: (-(d["drag"] if d["drag"] is not None else 0.0), -d["issue_rate"]))
    return rows


def slope(xs, ys):
    """Least-squares slope of ys over xs (None if fewer than 2 distinct xs)."""
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len({x for x, _ in pairs}) < 2:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    den = sum((x - mx) ** 2 for x, _ in pairs)
    return sum((x - mx) * (y - my) for x, y in pairs) / den if den else None


def depth_trend(records):
    """Per-depth mean grammar and issue rate, plus the grammar slope over depth."""
    groups = defaultdict(list)
    for r in records:
        if r.get("depth") is not None:
            groups[r["depth"]].append(r)
    rows = []
    for depth in sorted(groups):
        recs = groups[depth]
        rows.append(
            {
                "depth": depth,
                "n": len(recs),
                "mean_grammar": _mean([r.get("grammar_score") for r in recs]),
                "issue_rate": sum(1 for r in recs if detect_issues(r)) / len(recs),
            }
        )
    xs = [row["depth"] for row in rows]
    return {
        "rows": rows,
        "grammar_slope": slope(xs, [row["mean_grammar"] for row in rows]),
        "issue_slope": slope(xs, [row["issue_rate"] for row in rows]),
    }


def judge_gap(records):
    """Per-category gap between local grammar_score and GPT judge_score.

    Only records carrying both fields are used. gap = local - judge; a
    positive gap means the local verifier is more lenient than the judge.
    """
    groups = defaultdict(list)
    for r in records:
        if r.get("grammar_score") is not None and r.get("judge_score") is not None:
            groups[r.get("category")].append(r)
    rows = []
    for category, recs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        local = _mean([r["grammar_score"] for r in recs])
        judge = _mean([r["judge_score"] for r in recs])
        rows.append(
            {"category": category, "n": len(recs), "local": local, "judge": judge, "gap": local - judge}
        )
    return rows


def recommendations(records):
    """Concrete recommendations derived from the observed data."""
    if not records:
        return []
    recs = []
    for item in rank_issues(records):
        recs.append(f"{item['description']} ({item['rate']:.1%} of replies): {item['fix']}")
    trend = depth_trend(records)
    if trend["grammar_slope"] is not None and trend["grammar_slope"] < -DEPTH_SLOPE_THRESHOLD:
        recs.append(
            "Grammar declines with dialogue depth: keep the prompt's language instruction "
            "in every turn (not only the system prompt), trim or summarize old history, "
            "and prefer shorter replies in later turns."
        )
    elif trend["issue_slope"] is not None and trend["issue_slope"] > DEPTH_SLOPE_THRESHOLD:
        recs.append(
            "Issue rate rises with dialogue depth: re-inject the Sanskrit-only instruction "
            "each turn and run the verifier-in-the-loop retry on every turn."
        )
    cats = [c for c in by_category(records) if c["drag"] is not None and c["drag"] > 0][:3]
    if cats:
        names = ", ".join(str(c["category"]) for c in cats)
        recs.append(
            f"Focus prompt examples and regression tests on the weakest categories: {names}."
        )
    gaps = judge_gap(records)
    if gaps:
        avg = _mean([g["gap"] for g in gaps])
        if avg is not None and abs(avg) > 0.1:
            direction = "more lenient" if avg > 0 else "harsher"
            recs.append(
                f"The local grammar verifier is {direction} than the GPT5.6 judge by "
                f"{abs(avg):.2f} on average: recalibrate its threshold before using it "
                "as the retry trigger."
            )
    return recs


def _f(value):
    return "-" if value is None else f"{value:.3f}"


def render_report(records):
    """Render the analysis as markdown."""
    if not records:
        return "No records to analyze.\n"
    out = [f"# Coherence analysis ({len(records)} replies)", ""]

    out += ["## Issues ranked by ease of fix", ""]
    ranked = rank_issues(records)
    if ranked:
        out += ["| ease | issue | rate |", "| --- | --- | --- |"]
        out += [f"| {i['ease']} | {i['description']} | {i['rate']:.1%} |" for i in ranked]
    else:
        out.append("No issues detected with the current thresholds.")
    out.append("")

    out += ["## Components dragging quality down", ""]
    out += ["| category | n | mean grammar | drag | issue rate |", "| --- | --- | --- | --- | --- |"]
    for c in by_category(records):
        out.append(
            f"| {c['category']} | {c['n']} | {_f(c['mean_grammar'])} | {_f(c['drag'])} "
            f"| {c['issue_rate']:.1%} |"
        )
    out.append("")

    trend = depth_trend(records)
    out += ["## Depth trend", ""]
    out += ["| depth | n | mean grammar | issue rate |", "| --- | --- | --- | --- |"]
    for row in trend["rows"]:
        out.append(
            f"| {row['depth']} | {row['n']} | {_f(row['mean_grammar'])} | {row['issue_rate']:.1%} |"
        )
    out += [
        "",
        f"Grammar slope per depth: {_f(trend['grammar_slope'])}; "
        f"issue-rate slope per depth: {_f(trend['issue_slope'])}",
        "",
    ]

    gaps = judge_gap(records)
    out += ["## Local verifier vs GPT5.6 judge", ""]
    if gaps:
        out += ["| category | n | local | judge | gap |", "| --- | --- | --- | --- | --- |"]
        out += [
            f"| {g['category']} | {g['n']} | {_f(g['local'])} | {_f(g['judge'])} | {_f(g['gap'])} |"
            for g in gaps
        ]
    else:
        out.append("No records carry both `grammar_score` and `judge_score`.")
    out.append("")

    out += ["## Recommendations", ""]
    recs = recommendations(records)
    out += [f"{i}. {r}" for i, r in enumerate(recs, 1)] or ["None."]
    out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MITRA coherence result analysis")
    parser.add_argument("--path", help="JSONL records path (default: env or logs/)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rep = sub.add_parser("report", help="print the analysis report")
    rep.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = parser.parse_args(argv)

    records = load_records(args.path)
    if args.json:
        payload = {
            "n": len(records),
            "ranked_issues": rank_issues(records),
            "categories": by_category(records),
            "depth_trend": depth_trend(records),
            "judge_gap": judge_gap(records),
            "recommendations": recommendations(records),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(render_report(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
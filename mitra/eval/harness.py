"""Test-harness logger for the MITRA coherence check.

Stores every Reachy Mini reply as a structured record so that per-category,
per-depth scores can be recomputed later.

Record fields:
    ts             UTC ISO-8601 timestamp
    category       question category / component under test
    seed_question  the seed question that started the dialogue
    run            run number (int)
    depth          dialogue depth: first spoken line = 0, nth spoken line = n-1
    text           raw reply text (UTF-8, not escaped)
    grammar_score  float from the grammar verifier (None if not run)
    hindi_flag     bool from the Hindi detector (None if not run)
    verifier_error optional string if a verifier raised

Usage (from the mitra/ directory):

    from eval.harness import Harness

    harness = Harness(grammar_fn=my_grammar_scorer, hindi_fn=my_hindi_detector)
    conv = harness.conversation(category="greeting", seed_question="...", run=1)
    conv.log_line(reply_text)      # depth 0
    conv.log_line(next_reply)      # depth 1

    python -m eval.harness summary            # per-category, per-depth table
    python -m eval.harness csv out.csv        # export JSONL to CSV

The output path defaults to mitra/logs/harness_records.jsonl and can be
overridden with the MITRA_HARNESS_LOG environment variable.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "harness_records.jsonl"

FIELDS = [
    "ts",
    "category",
    "seed_question",
    "run",
    "depth",
    "text",
    "grammar_score",
    "hindi_flag",
    "verifier_error",
]


def _log_path(path=None) -> Path:
    if path:
        return Path(path)
    override = os.environ.get("MITRA_HARNESS_LOG")
    return Path(override) if override else DEFAULT_LOG_PATH


class Conversation:
    """One dialogue (category, seed question, run) with an automatic depth counter."""

    def __init__(self, harness, category, seed_question, run):
        self._harness = harness
        self.category = category
        self.seed_question = seed_question
        self.run = run
        self._next_depth = 0

    def log_line(self, text, **extra):
        """Record the next spoken line; depth is 0 for the first, n-1 for the nth."""
        record = self._harness.record(
            category=self.category,
            seed_question=self.seed_question,
            run=self.run,
            depth=self._next_depth,
            text=text,
            **extra,
        )
        self._next_depth += 1
        return record


class Harness:
    """Writes structured records, optionally running verifier callables.

    grammar_fn(text) -> float   grammar score
    hindi_fn(text)   -> bool    True if the text is (mis)spoken in Hindi
    Either may be None, in which case the field is stored as null.
    """

    def __init__(self, grammar_fn=None, hindi_fn=None, path=None):
        self.grammar_fn = grammar_fn
        self.hindi_fn = hindi_fn
        self.path = _log_path(path)

    def conversation(self, category, seed_question, run):
        return Conversation(self, category, seed_question, run)

    def record(self, category, seed_question, run, depth, text, **extra):
        errors = []
        grammar_score = None
        hindi_flag = None
        if self.grammar_fn is not None:
            try:
                grammar_score = self.grammar_fn(text)
            except Exception as exc:  # keep the run going; store the failure
                errors.append(f"grammar_fn: {exc}")
        if self.hindi_fn is not None:
            try:
                hindi_flag = bool(self.hindi_fn(text))
            except Exception as exc:
                errors.append(f"hindi_fn: {exc}")

        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "category": category,
            "seed_question": seed_question,
            "run": run,
            "depth": depth,
            "text": text,
            "grammar_score": grammar_score,
            "hindi_flag": hindi_flag,
        }
        if errors:
            rec["verifier_error"] = "; ".join(errors)
        rec.update(extra)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


def load_records(path=None):
    """Return all harness records as a list of dicts."""
    p = _log_path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def export_csv(out_path, path=None):
    """Convert the JSONL records to CSV (extra fields are appended as columns)."""
    records = load_records(path)
    extra_fields = sorted({k for r in records for k in r if k not in FIELDS})
    columns = FIELDS + extra_fields
    with Path(out_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow({c: r.get(c) for c in columns})
    return len(records)


def summarize(records):
    """Per (category, depth): count, mean grammar_score, Hindi rate."""
    groups = defaultdict(list)
    for r in records:
        groups[(r.get("category"), r.get("depth"))].append(r)
    rows = []
    for (category, depth), recs in sorted(
        groups.items(), key=lambda kv: (str(kv[0][0]), kv[0][1] if kv[0][1] is not None else -1)
    ):
        scores = [r["grammar_score"] for r in recs if r.get("grammar_score") is not None]
        flags = [r["hindi_flag"] for r in recs if r.get("hindi_flag") is not None]
        rows.append(
            {
                "category": category,
                "depth": depth,
                "n": len(recs),
                "mean_grammar_score": sum(scores) / len(scores) if scores else None,
                "hindi_rate": sum(1 for x in flags if x) / len(flags) if flags else None,
            }
        )
    return rows


def _fmt(value):
    return "-" if value is None else f"{value:.3f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="MITRA coherence harness records")
    parser.add_argument("--path", help="JSONL records path (default: env or logs/)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary", help="print per-category, per-depth summary")
    csv_p = sub.add_parser("csv", help="export records to CSV")
    csv_p.add_argument("out", help="output CSV path")
    args = parser.parse_args(argv)

    if args.cmd == "summary":
        records = load_records(args.path)
        if not records:
            print("No records at", _log_path(args.path))
            return 0
        print(f"{'category':<20} {'depth':>5} {'n':>4} {'grammar':>8} {'hindi':>6}")
        for row in summarize(records):
            print(
                f"{str(row['category']):<20} {row['depth']:>5} {row['n']:>4} "
                f"{_fmt(row['mean_grammar_score']):>8} {_fmt(row['hindi_rate']):>6}"
            )
        return 0

    if args.cmd == "csv":
        count = export_csv(args.out, args.path)
        print(f"Wrote {count} records to {args.out}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Retrieval precision harness for the phrasebook (DESIGN §4).

The phrasebook is the accuracy mechanism for conversational Sanskrit: it hands
the model human-authored, grammatically correct sentences to write in the style
of. When it retrieves the wrong rows the model is actively misled, so retrieval
quality is measurable on its own, without a Sanskrit grader in the loop.

Labelled set: tests/fixtures/retrieval_eval.jsonl, drawn from real transcripts in
logs/turns.jsonl (Whisper's actual spellings, misspellings included — romanised
Sanskrit is the dominant input, not English). ``gold`` lists the row ids that
would genuinely help answer that turn; an empty ``gold`` is a NEGATIVE case
where the corpus has nothing relevant and the correct behaviour is to attach
nothing at all.

    python3 scripts/eval_retrieval.py            # score current retrieval
    python3 scripts/eval_retrieval.py -v         # show per-query rows
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import mitra  # noqa: F401
except ImportError:
    spec = importlib.util.spec_from_file_location(
        "mitra", ROOT / "src" / "__init__.py",
        submodule_search_locations=[str(ROOT / "src")])
    module = importlib.util.module_from_spec(spec)
    sys.modules["mitra"] = module
    spec.loader.exec_module(module)

EVAL_PATH = ROOT / "tests" / "fixtures" / "retrieval_eval.jsonl"
PHRASEBOOK_PATH = ROOT / "data" / "phrasebook.jsonl"


def load_cases(path: Path) -> list[dict]:
    cases = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def evaluate(pb, cases: list[dict], k: int = 3, verbose: bool = False) -> dict:
    """Precision/recall over positives, plus noise rate over negatives.

    ``hit_rate`` is the headline number: the share of answerable turns where at
    least one genuinely relevant row reached the model. ``noise_rate`` is its
    counterpart on unanswerable turns — how often the model was handed
    irrelevant Sanskrit and told to imitate it.
    """
    pos = [c for c in cases if c["gold"]]
    neg = [c for c in cases if not c["gold"]]
    hits = 0
    precision_sum = 0.0
    noisy = 0

    for case in cases:
        rows = pb.similar(case["query"], k=k)
        got = [r["id"] for r in rows]
        gold = set(case["gold"])
        relevant = [rid for rid in got if rid in gold]

        if gold:
            if relevant:
                hits += 1
            precision_sum += len(relevant) / len(got) if got else 0.0
        elif got:
            noisy += 1

        if verbose:
            mark = "-" if not gold else ("HIT " if relevant else "MISS")
            if not gold:
                mark = "NOISE" if got else "ok   "
            print(f"[{mark}] {case['query']!r}")
            for r in rows:
                flag = "*" if r["id"] in gold else " "
                print(f"      {flag} {r['id']} | {r['english']} -> {r['sanskrit']}")
            if not rows:
                print("        (nothing attached)")

    return {
        "positives": len(pos),
        "negatives": len(neg),
        "hit_rate": hits / len(pos) if pos else 0.0,
        "precision": precision_sum / len(pos) if pos else 0.0,
        "noise_rate": noisy / len(neg) if neg else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    from mitra.lexicon.phrasebook import Phrasebook

    pb = Phrasebook(PHRASEBOOK_PATH)
    if not pb.count():
        print(f"phrasebook empty at {PHRASEBOOK_PATH}", file=sys.stderr)
        return 1

    cases = load_cases(EVAL_PATH)
    m = evaluate(pb, cases, k=args.k, verbose=args.verbose)

    print(f"\ncorpus {pb.count()} rows | {m['positives']} answerable "
          f"+ {m['negatives']} unanswerable queries\n")
    print(f"  hit rate    {m['hit_rate']:6.1%}   (answerable turns with >=1 relevant row)")
    print(f"  precision   {m['precision']:6.1%}   (share of attached rows that are relevant)")
    print(f"  noise rate  {m['noise_rate']:6.1%}   (unanswerable turns handed irrelevant rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

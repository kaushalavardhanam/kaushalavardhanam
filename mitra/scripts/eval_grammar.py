#!/usr/bin/env python3
"""Measure the Sanskrit checks before they are allowed to reject anything.

Two corpora, two different questions:

* ``data/phrasebook.jsonl`` — 981 human-authored sentences. Everything the
  checks flag here is a FALSE POSITIVE, and a false positive costs a retry and
  possibly the safe fallback, i.e. it makes Mitra go quiet mid-conversation.
  The vocabulary is deliberately built WITHOUT the phrasebook for this run,
  so the sentences are unseen; the runtime build does include it.
* ``logs/turns.jsonl`` — every reply Mitra has actually spoken. Flags here are
  what the checks are for. This is not a precision measure (the log has no
  gold labels), so the script prints the flagged sentences for reading.

    python3 scripts/eval_grammar.py
    python3 scripts/eval_grammar.py -v        # show every flagged sentence
"""

from __future__ import annotations

import argparse
import collections
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

from mitra.lexicon.vocabulary import Vocabulary  # noqa: E402
from mitra.sanskrit import Analyzer  # noqa: E402
from mitra.sanskrit import grammar  # noqa: E402


def load_phrasebook() -> list[str]:
    path = ROOT / "data" / "phrasebook.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        sentence = row.get("sanskrit", "")
        # Single-word rows are vocabulary entries, not sentences; the checks
        # are about how words combine.
        if sentence and len(sentence.split()) >= 2:
            out.append(sentence)
    return out


def load_replies() -> list[str]:
    path = ROOT / "logs" / "turns.jsonl"
    if not path.exists():
        return []
    seen, out = set(), []
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        reply = (row.get("reply") or "").strip()
        if reply and reply not in seen:
            seen.add(reply)
            out.append(reply)
    return out


def run(name: str, sentences: list[str], analyzer, vocabulary,
        verbose: bool) -> dict:
    counts = collections.Counter()
    words = collections.Counter()
    flagged = []
    for sentence in sentences:
        findings = grammar.check(sentence, analyzer, vocabulary)
        if findings:
            flagged.append((sentence, findings))
            for finding in findings:
                counts[finding.check] += 1
                words.update(finding.words)
    total = len(sentences) or 1
    print(f"\n{name}: {len(flagged)}/{len(sentences)} sentences flagged "
          f"= {len(flagged)/total:.1%}")
    for check, n in counts.most_common():
        print(f"    {check:14} {n}")
    print("    most-flagged words: " +
          ", ".join(f"{w}({n})" for w, n in words.most_common(12)))
    if verbose:
        for sentence, findings in flagged:
            print(f"    ✗ {sentence}")
            for finding in findings:
                print(f"        {finding.check}: {finding.detail}")
    return {"flagged": len(flagged), "total": len(sentences)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    analyzer = Analyzer(ROOT / "data" / "vidyut")
    if not analyzer.available:
        print("kosha unavailable — run scripts/fetch_sanskrit_data.py", file=sys.stderr)
        return 1
    vocabulary = Vocabulary(
        analyzer, seed_path=ROOT / "src" / "lexicon" / "seed_lexicon.json")

    phrasebook = load_phrasebook()
    replies = load_replies()

    if phrasebook:
        # Held-out split: the runtime vocabulary absorbs the phrasebook, so
        # scoring it against a vocabulary built from the whole corpus would
        # measure nothing. Half is absorbed, the other half stands in for
        # correct Sanskrit the whitelist has never seen — which is the case
        # that decides whether these checks can be allowed to reject.
        absorbed, held_out = phrasebook[::2], phrasebook[1::2]
        seen_vocabulary = Vocabulary(
            analyzer, seed_path=ROOT / "src" / "lexicon" / "seed_lexicon.json",
            extra_texts=tuple(absorbed))
        run("word list only, phrasebook UNSEEN (strictest possible config)",
            held_out, analyzer, vocabulary, args.verbose)
        run("word list + half the phrasebook, scored on the other half "
            "(≈ the shipped config)", held_out, analyzer, seen_vocabulary,
            args.verbose)
    else:
        print("phrasebook absent — skipping the false-positive measurement")

    if replies:
        full = Vocabulary(
            analyzer, seed_path=ROOT / "src" / "lexicon" / "seed_lexicon.json",
            extra_texts=tuple(phrasebook))
        run("Mitra's own replies (flags are what the checks are for)",
            replies, analyzer, full, args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

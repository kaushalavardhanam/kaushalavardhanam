#!/usr/bin/env python3
"""Index the Cologne dictionaries for word-choice lookups (DESIGN §5).

Two directions, two different jobs:

  Monier-Williams (mw.txt)  Sanskrit → English. What does this word actually
                            mean? Used to gloss whitelist entries for review,
                            and to check that a word Mitra is about to teach a
                            child means what the word list claims.
  Apte (ae.txt)             English → Sanskrit. This is the one that improves
                            word choice at runtime: asked for "butter", Apte
                            answers नवनीतम्, which is exactly the word the
                            model failed to produce when it said मक्खनम्.

The source files are Cologne's own markup (``<L>`` … ``<LEND>`` records with
SLP1 in ``<s>`` tags). They are parsed once into a SQLite file so lookups cost
a single indexed query instead of a 48 MB scan.

    python3 scripts/fetch_sanskrit_data.py --cologne
    python3 scripts/build_dictionary.py
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CDSL_DIR = ROOT / "data" / "cdsl"
DB_PATH = CDSL_DIR / "cologne.db"

_RECORD = re.compile(r"<L>(.*?)<LEND>", re.S)
_KEY = re.compile(r"<k1>([^<]*)")
_SLP = re.compile(r"<s>([^<]*)</s>")
_TAG = re.compile(r"<[^>]*>")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mw (key TEXT, meaning TEXT);
CREATE TABLE IF NOT EXISTS ae (word TEXT, sanskrit TEXT, meaning TEXT);
CREATE INDEX IF NOT EXISTS mw_key ON mw(key);
CREATE INDEX IF NOT EXISTS ae_word ON ae(word);
"""


def plain(body: str) -> str:
    """Record body with the markup stripped, whitespace collapsed."""
    text = _TAG.sub(" ", body).replace("¦", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_mw(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    for record in _RECORD.findall(text):
        key = _KEY.search(record)
        if not key:
            continue
        body = record.split(">", 1)[-1]
        meaning = plain(body)
        if meaning:
            yield key.group(1).strip(), meaning[:2000]


def parse_ae(path: Path):
    """Apte's English headword → the SLP1 equivalents it offers, in order."""
    text = path.read_text(encoding="utf-8", errors="replace")
    # Apte's records are not all <LEND>-terminated, so split on the next <L>.
    for record in re.split(r"(?=<L>\d)", text):
        key = _KEY.search(record)
        if not key:
            continue
        # Each <s> group is a comma-separated list of candidates; keep their
        # order — Apte puts the ordinary word first and the poetic ones after.
        candidates: list[str] = []
        for group in _SLP.findall(record):
            for word in group.split(","):
                word = word.strip()
                if word and word not in candidates:
                    candidates.append(word)
        if candidates:
            yield (key.group(1).strip().lower(), ",".join(candidates[:12]),
                   plain(record)[:1000])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DB_PATH))
    args = ap.parse_args()

    mw_path, ae_path = CDSL_DIR / "mw.txt", CDSL_DIR / "ae.txt"
    if not mw_path.exists() or not ae_path.exists():
        print("dictionary sources missing — run "
              "scripts/fetch_sanskrit_data.py --cologne", file=sys.stderr)
        return 1

    out = Path(args.out)
    if out.exists():
        out.unlink()
    db = sqlite3.connect(out)
    db.executescript(_SCHEMA)
    with db:
        rows = list(parse_mw(mw_path))
        db.executemany("INSERT INTO mw VALUES (?, ?)", rows)
        print(f"  Monier-Williams: {len(rows)} entries")
        rows = list(parse_ae(ae_path))
        db.executemany("INSERT INTO ae VALUES (?, ?, ?)", rows)
        print(f"  Apte English-Sanskrit: {len(rows)} entries")
    db.close()
    print(f"{out.relative_to(ROOT)} ({out.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

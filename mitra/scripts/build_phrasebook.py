#!/usr/bin/env python3
"""Build data/phrasebook.jsonl from संस्कृत व्यवहार साहस्री (daily.pdf).

    python3 scripts/build_phrasebook.py path/to/daily.pdf
    python3 scripts/build_phrasebook.py path/to/daily.pdf --report

The source is ~1000 everyday sentences across 27 situations, each line shaped
``<devanagari> = <english>``. Three things stand between that and a usable
retrieval corpus:

1. The text layer duplicates fragments — ``सुप्रभातम् ।सुप्रभातम्``,
   ``भवान् किंभवान् किं``, ``अन्नं बहु उष्णम्।उष्णम्``. These come from
   overlapping text runs (emphasis layered over base text), not from the book.
   Indexed as-is, every retrieval hit would return garbled Sanskrit and the
   TTS would read the duplicate aloud.

2. There is no IAST column, and we need one. Transcripts arrive romanised
   ("Namaste. Katamasi.", "Aham kushali asmi") because Whisper tags spoken
   Sanskrit as English. Matching those against an English gloss scores near
   zero; matching against IAST scores high. Without this field, retrieval
   misses on exactly the turns where the user speaks Sanskrit.

3. Chapter headings share the ``=`` shape with phrase lines and have to be
   told apart, so each row can carry its situation.

LICENCE: the book is published and copyrighted by Pallava Prakashan and the
transliteration is marked for personal study and research only. Add
``data/phrasebook.jsonl`` to .gitignore — ship this script, not its output.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Devanagari digits open a chapter heading: "१. शिष्टाचारः = Common formulas"
_CHAPTER_RE = re.compile(r"^([\u0966-\u096F]+)\s*[.।]\s*(.+)$")

# Lines that are page furniture rather than content.
_NOISE = re.compile(
    r"^(daily\.pdf|sanskritdocuments\.org|संस्कृत व्यवहार साहस्री|\d+|"
    r"One Thousand Sentences.*|.*Pallava Prakashan.*|.*Aksharam.*|"
    r"Encoded and proofread.*|.*sanskrit@cheerful.*)$",
    re.IGNORECASE,
)

# A usable row needs Devanagari on the left of an '='.
_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


# --------------------------------------------------------------- extraction

def extract_text(pdf: Path) -> str:
    """Prefer pdftotext -layout; fall back to pdfplumber.

    The two extractors disagree about the duplication artifact, so if one
    produces clean text the dedup pass below becomes a no-op. Worth checking
    --report to see which you got.
    """
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"],
            capture_output=True, check=True,
        )
        text = out.stdout.decode("utf-8", errors="replace")
        if text.strip():
            return text
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    try:
        import pdfplumber
    except ImportError:
        sys.exit("Need either poppler-utils (pdftotext) or pdfplumber.\n"
                 "  sudo apt install poppler-utils\n"
                 "  pip install pdfplumber")

    parts = []
    with pdfplumber.open(str(pdf)) as doc:
        for page in doc.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


# ------------------------------------------------------------------- dedup

def _collapse_once(s: str) -> str:
    """Remove one trailing repeated chunk, with or without a danda between.

    Longest-first so ``भवान् किंभवान् किं`` collapses as a unit rather than
    leaving a fragment behind.
    """
    n = len(s)
    for size in range(n // 2, 1, -1):
        tail = s[n - size:]
        head = s[:n - size]
        if head.endswith(tail):                       # X X
            return head
        m = re.match(r"^(.*?" + re.escape(tail) + r")\s*।\s*$", head)
        if m:                                          # X ।X
            return m.group(1) + " ।"
    return s


def dedup(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    for _ in range(4):                                 # some lines double twice
        collapsed = _collapse_once(s)
        if collapsed == s:
            break
        s = collapsed
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------- transliteration

def make_transliterator():
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except ImportError:
        print("WARNING: indic-transliteration not installed — iast will be "
              "empty, which cripples retrieval on romanised transcripts.\n"
              "  pip install indic-transliteration", file=sys.stderr)
        return lambda _text: ""

    def to_iast(text: str) -> str:
        try:
            return transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
        except Exception:
            return ""

    return to_iast


# ------------------------------------------------------------------ parsing

def parse(text: str, to_iast) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    changed: list[dict] = []
    chapter_sa = chapter_en = ""
    chapter_no = "00"
    seen: set[str] = set()

    for raw in text.splitlines():
        line = raw.strip()
        if not line or _NOISE.match(line):
            continue
        if "=" not in line:
            continue

        left, _, right = line.partition("=")
        left, right = left.strip(), right.strip()
        if not left or not right or not _DEVANAGARI.search(left):
            continue

        heading = _CHAPTER_RE.match(left)
        if heading:
            chapter_no = str(len(rows) and chapter_no or "00")
            chapter_sa = dedup(heading.group(2))
            chapter_en = right
            chapter_no = f"{_devanagari_int(heading.group(1)):02d}"
            continue

        cleaned = dedup(left)
        if cleaned != left:
            changed.append({"before": left, "after": cleaned})
        if cleaned in seen:
            continue
        seen.add(cleaned)

        rows.append({
            "id": f"{chapter_no}-{len(rows) + 1:04d}",
            "chapter": chapter_sa,
            "chapter_en": chapter_en,
            "sanskrit": cleaned,
            "iast": to_iast(cleaned),
            "english": right,
        })
    return rows, changed


def _devanagari_int(s: str) -> int:
    return int("".join(str(ord(c) - 0x0966) for c in s) or "0")


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("data/phrasebook.jsonl"))
    ap.add_argument("--report", action="store_true",
                    help="print stats and sample rows instead of a bare count")
    args = ap.parse_args()

    if not args.pdf.exists():
        return f"not found: {args.pdf}"

    text = extract_text(args.pdf)
    rows, changed = parse(text, make_transliterator())
    if not rows:
        return ("Parsed 0 rows. The extractor may have produced a different "
                "line shape than '<devanagari> = <english>' — inspect with:\n"
                f"  pdftotext -layout {args.pdf} - | head -60")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Everything the dedup pass touched, for a human to skim. This is the file
    # to hand your Sanskrit reviewer — it is short and it is where the risk is.
    review = args.out.with_name(args.out.stem + "_review.tsv")
    with review.open("w", encoding="utf-8") as fh:
        fh.write("before\tafter\n")
        for item in changed:
            fh.write(f"{item['before']}\t{item['after']}\n")

    print(f"wrote {len(rows)} rows → {args.out}")
    print(f"dedup touched {len(changed)} lines → {review}")

    if args.report:
        chapters: dict[str, int] = {}
        no_iast = 0
        for row in rows:
            chapters[row["chapter_en"] or "(none)"] = \
                chapters.get(row["chapter_en"] or "(none)", 0) + 1
            if not row["iast"]:
                no_iast += 1
        print(f"\nchapters: {len(chapters)}")
        for name, count in list(chapters.items())[:30]:
            print(f"  {count:4d}  {name}")
        if no_iast:
            print(f"\nWARNING: {no_iast} rows have no IAST — retrieval on "
                  "romanised transcripts will miss these.")
        print("\nsample:")
        for row in rows[:8]:
            print(f"  {row['sanskrit']}")
            print(f"    {row['iast']}")
            print(f"    {row['english']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

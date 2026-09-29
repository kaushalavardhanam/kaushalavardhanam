#!/usr/bin/env python3
"""Download the linguistic data the Sanskrit checks need (DESIGN §5).

Two datasets, both offline once fetched, both out of git (data/ is ignored):

  data/vidyut/   ~78 MB  vidyut 0.4.0 — a 30M-form inflected lexicon with
                         Pāṇinian morphology, plus a segmenter and a
                         derivation engine. MIT licensed, ambuda-org.
  data/cdsl/     ~55 MB  Cologne Digital Sanskrit Lexicon sources: Monier-
                         Williams (Sanskrit→English) and Apte (English→
                         Sanskrit), CC-BY-SA-4.0, from sanskrit-lexicon/csl-orig.

    python3 scripts/fetch_sanskrit_data.py            # both
    python3 scripts/fetch_sanskrit_data.py --vidyut   # just the morphology
    python3 scripts/fetch_sanskrit_data.py --cologne  # just the dictionaries
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VIDYUT_URL = ("https://github.com/ambuda-org/vidyut/releases/download/"
              "py-0.4.0/data-0.4.0.zip")
VIDYUT_DIR = ROOT / "data" / "vidyut"

# Raw dictionary sources. The pycdsl package would normally fetch these, but
# its downloader scrapes a Cologne page that no longer exists (it fails with
# AttributeError on a missing footer), so the files are taken from the
# project's own git mirror instead.
COLOGNE_FILES = {
    "mw.txt": "https://raw.githubusercontent.com/sanskrit-lexicon/csl-orig/master/v02/mw/mw.txt",
    "ae.txt": "https://raw.githubusercontent.com/sanskrit-lexicon/csl-orig/master/v02/ae/ae.txt",
}
COLOGNE_DIR = ROOT / "data" / "cdsl"


def _download(url: str, note: str) -> bytes:
    print(f"  fetching {note} …", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "mitra-setup"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return response.read()


def fetch_vidyut() -> None:
    if (VIDYUT_DIR / "kosha").exists():
        print(f"vidyut data already present at {VIDYUT_DIR}")
        return
    print("vidyut morphology data (~32 MB download, ~78 MB on disk)")
    payload = _download(VIDYUT_URL, "data-0.4.0.zip")
    VIDYUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(VIDYUT_DIR)
    print(f"  unpacked to {VIDYUT_DIR}")


def fetch_cologne() -> None:
    COLOGNE_DIR.mkdir(parents=True, exist_ok=True)
    print("Cologne dictionaries (~55 MB)")
    for name, url in COLOGNE_FILES.items():
        target = COLOGNE_DIR / name
        if target.exists():
            print(f"  {name} already present")
            continue
        target.write_bytes(_download(url, name))
        print(f"  saved {target} ({target.stat().st_size / 1e6:.0f} MB)")
    print("  build the index with: python3 scripts/build_dictionary.py")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vidyut", action="store_true")
    parser.add_argument("--cologne", action="store_true")
    args = parser.parse_args()
    both = not (args.vidyut or args.cologne)
    try:
        if both or args.vidyut:
            fetch_vidyut()
        if both or args.cologne:
            fetch_cologne()
    except OSError as e:
        print(f"download failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

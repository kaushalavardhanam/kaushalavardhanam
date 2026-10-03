"""Fetch and sample conversation starters for the MITRA coherence check.

Source: https://sanskritdocuments.org/doc_z_misc_major_works/daily.html

The page is parsed into categories (the issue expects 27). Within each
category, up to 3 questions are chosen at random using a fixed, recorded seed,
and the selection is written to a versioned JSON file. Each selected question
is meant to be tested RUNS_PER_QUESTION (3) times with the model.

Usage (from the mitra/ directory):

    python -m eval.starters fetch                 # download, parse, sample, save
    python -m eval.starters fetch --html daily.html   # parse a saved copy instead
    python -m eval.starters show                  # print the saved selection

The page structure could not be checked when this was written, so the parser
is heuristic:
  * a category starts at each heading tag (default h2,h3; see --heading-tags)
  * a candidate question is any text line in a category that contains "?"
If the number of categories found is not 27 the command reports it and exits
non-zero (unless --allow-mismatch); adjust --heading-tags and re-run.

Output: mitra/data/conversation_starters_<version>.json. An existing file is
never overwritten unless --force is given; bump --version for a new selection.
"""

import argparse
import hashlib
import json
import random
import re
import sys
import urllib.request
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

SOURCE_URL = "https://sanskritdocuments.org/doc_z_misc_major_works/daily.html"
EXPECTED_CATEGORIES = 27
MAX_QUESTIONS_PER_CATEGORY = 3
RUNS_PER_QUESTION = 3
SEED = 13
DEFAULT_VERSION = "v1"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

_BLOCK_TAGS = {
    "p", "div", "li", "ul", "ol", "tr", "td", "th", "table", "br", "pre",
    "blockquote", "dd", "dt", "dl", "section", "article", "hr",
}
_SKIP_TAGS = {"script", "style"}


def output_path(version=DEFAULT_VERSION):
    return DATA_DIR / f"conversation_starters_{version}.json"


def _clean(text):
    return re.sub(r"\s+", " ", text).strip()


class _PageParser(HTMLParser):
    """Collects (category name -> list of text lines) from headings and blocks."""

    def __init__(self, heading_tags, split_newlines=False):
        super().__init__(convert_charrefs=True)
        self.heading_tags = set(heading_tags)
        self.split_newlines = split_newlines
        self.categories = []  # list of [name, [lines]]
        self._buf = []
        self._heading = None  # tag name while inside a heading
        self._skip = 0
        self._pre = 0

    def _flush(self):
        text = _clean("".join(self._buf))
        self._buf = []
        if not text:
            return
        if self._heading is not None:
            return  # heading text is handled in handle_endtag
        if self.categories:
            self.categories[-1][1].append(text)

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
            return
        if tag in self.heading_tags:
            self._flush()
            self._heading = tag
            return
        if tag == "pre":
            self._pre += 1
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
            return
        if tag == self._heading:
            name = _clean("".join(self._buf))
            self._buf = []
            self._heading = None
            if name:
                self.categories.append([name, []])
            return
        if tag == "pre":
            self._flush()
            self._pre = max(0, self._pre - 1)
            return
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data):
        if self._skip:
            return
        if self._heading is None and (self._pre or self.split_newlines) and "\n" in data:
            parts = data.split("\n")
            for i, part in enumerate(parts):
                self._buf.append(part)
                if i < len(parts) - 1:
                    self._flush()
            return
        self._buf.append(data)

    def close(self):
        super().close()
        self._flush()


def fetch_html(url=SOURCE_URL, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "mitra-coherence-check/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def parse_categories(html, heading_tags=("h2", "h3"), split_newlines=False):
    """Return an ordered list of {"name", "questions"} dicts (questions deduped)."""
    parser = _PageParser(heading_tags, split_newlines=split_newlines)
    parser.feed(html)
    parser.close()
    result = []
    for name, lines in parser.categories:
        seen = set()
        questions = []
        for line in lines:
            if ("?" in line or "？" in line) and len(line) >= 3 and line not in seen:
                seen.add(line)
                questions.append(line)
        result.append({"name": name, "questions": questions})
    return result


def sample_questions(categories, seed=SEED, k=MAX_QUESTIONS_PER_CATEGORY):
    """Pick up to k questions per category.

    Each category has its own RNG derived from (seed, category index), so the
    selection is reproducible and independent of other categories' contents.
    Selected questions are kept in document order.
    """
    out = []
    for idx, cat in enumerate(categories):
        pool = cat["questions"]
        rng = random.Random(f"{seed}:{idx}")
        chosen_idx = sorted(rng.sample(range(len(pool)), min(k, len(pool))))
        out.append(
            {
                "index": idx,
                "name": cat["name"],
                "n_available": len(pool),
                "questions": [
                    {"id": f"c{idx:02d}q{n}", "text": pool[i]}
                    for n, i in enumerate(chosen_idx)
                ],
            }
        )
    return out


def build_selection(html, version=DEFAULT_VERSION, seed=SEED, heading_tags=("h2", "h3"),
                    split_newlines=False, source=SOURCE_URL):
    categories = parse_categories(html, heading_tags, split_newlines)
    return {
        "version": version,
        "source_url": source,
        "source_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "heading_tags": list(heading_tags),
        "max_questions_per_category": MAX_QUESTIONS_PER_CATEGORY,
        "runs_per_question": RUNS_PER_QUESTION,
        "expected_categories": EXPECTED_CATEGORIES,
        "categories": sample_questions(categories, seed=seed),
    }


def save_selection(selection, path, force=False):
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; use a new --version or --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_selection(version=DEFAULT_VERSION, path=None):
    p = Path(path) if path else output_path(version)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_trials(selection):
    """Yield (category, seed_question, run) for every planned test run.

    Feed these to eval.harness.Harness.conversation(...); run is 1-based.
    """
    runs = selection.get("runs_per_question", RUNS_PER_QUESTION)
    for cat in selection["categories"]:
        for q in cat["questions"]:
            for run in range(1, runs + 1):
                yield cat["name"], q["text"], run


def _cmd_fetch(args):
    if args.html:
        html = Path(args.html).read_text(encoding="utf-8")
        source = f"{SOURCE_URL} (local copy: {Path(args.html).name})"
    else:
        html = fetch_html()
        source = SOURCE_URL
    tags = tuple(t.strip() for t in args.heading_tags.split(",") if t.strip())
    selection = build_selection(html, version=args.version, seed=args.seed,
                                heading_tags=tags, split_newlines=args.split_newlines,
                                source=source)
    cats = selection["categories"]
    total_q = sum(len(c["questions"]) for c in cats)
    print(f"Categories found: {len(cats)} (expected {EXPECTED_CATEGORIES})")
    print(f"Questions selected: {total_q}; test runs planned: {total_q * RUNS_PER_QUESTION}")
    for c in cats:
        if len(c["questions"]) < MAX_QUESTIONS_PER_CATEGORY:
            print(f"  note: '{c['name']}' has only {len(c['questions'])} question(s)")
    if len(cats) != EXPECTED_CATEGORIES and not args.allow_mismatch:
        print("Category count mismatch; nothing saved. Try --heading-tags or "
              "--split-newlines, or pass --allow-mismatch.", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else output_path(args.version)
    try:
        save_selection(selection, out, force=args.force)
    except FileExistsError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("Saved", out)
    return 0


def _cmd_show(args):
    sel = load_selection(args.version, args.out)
    print(f"version={sel['version']} seed={sel['seed']} source={sel['source_url']}")
    for c in sel["categories"]:
        print(f"[{c['index']:02d}] {c['name']} ({len(c['questions'])}/{c['n_available']})")
        for q in c["questions"]:
            print(f"     {q['id']}: {q['text']}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="MITRA conversation starters")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--out", help="override output path")
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="fetch, parse, sample and save")
    f.add_argument("--html", help="parse this saved HTML file instead of downloading")
    f.add_argument("--seed", type=int, default=SEED)
    f.add_argument("--heading-tags", default="h2,h3")
    f.add_argument("--split-newlines", action="store_true",
                   help="treat every newline in the HTML as a line break")
    f.add_argument("--allow-mismatch", action="store_true")
    f.add_argument("--force", action="store_true", help="overwrite an existing file")
    sub.add_parser("show", help="print the saved selection")
    args = parser.parse_args(argv)
    return _cmd_fetch(args) if args.cmd == "fetch" else _cmd_show(args)


if __name__ == "__main__":
    sys.exit(main())
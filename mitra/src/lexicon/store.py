"""SQLite lexicon of Sanskrit object names (FR-2.5/2.6, DESIGN §6).

Human-verified rows always override model generation — this is the primary
accuracy mechanism compensating for local-model Sanskrit (REQUIREMENTS R1/R2).
New model-generated names are stored ``unverified`` for later human review via
the ``mitra-lexicon`` CLI. An existing row is never overwritten by generation.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lexicon (
    object_en       TEXT PRIMARY KEY,
    name_devanagari TEXT NOT NULL,
    name_iast       TEXT,
    gloss_en        TEXT,
    verified        INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT NOT NULL
)
"""

SEED_PATH = Path(__file__).parent / "seed_lexicon.json"


def _norm(object_en: str) -> str:
    return object_en.strip().lower()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LexiconStore:
    def __init__(self, db_path: str | Path = ":memory:",
                 seed_path: Path | None = SEED_PATH):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(_SCHEMA)
        self._db.commit()
        if seed_path is not None and self.count() == 0:
            self.seed(seed_path)

    def count(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM lexicon").fetchone()[0]

    def seed(self, path: str | Path) -> int:
        entries = json.loads(Path(path).read_text(encoding="utf-8"))["entries"]
        with self._db:
            self._db.executemany(
                "INSERT OR IGNORE INTO lexicon VALUES (?, ?, ?, ?, 1, ?)",
                [(_norm(e["object_en"]), e["name_devanagari"], e.get("name_iast", ""),
                  e.get("gloss_en", e["object_en"]), _now()) for e in entries],
            )
        return self.count()

    def lookup(self, object_en: str) -> dict | None:
        row = self._db.execute(
            "SELECT * FROM lexicon WHERE object_en = ?", (_norm(object_en),)
        ).fetchone()
        return dict(row) if row else None

    def add_unverified(self, object_en: str, name_devanagari: str,
                       name_iast: str = "", gloss_en: str = "") -> bool:
        """Record a model-generated name for review. Never overwrites."""
        with self._db:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO lexicon VALUES (?, ?, ?, ?, 0, ?)",
                (_norm(object_en), name_devanagari, name_iast,
                 gloss_en or object_en, _now()),
            )
        return cur.rowcount == 1

    def verify(self, object_en: str, name_devanagari: str | None = None) -> None:
        with self._db:
            if name_devanagari:
                self._db.execute(
                    "UPDATE lexicon SET verified = 1, name_devanagari = ?, updated_at = ? "
                    "WHERE object_en = ?",
                    (name_devanagari, _now(), _norm(object_en)),
                )
            else:
                self._db.execute(
                    "UPDATE lexicon SET verified = 1, updated_at = ? WHERE object_en = ?",
                    (_now(), _norm(object_en)),
                )

    def pending_review(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM lexicon WHERE verified = 0 ORDER BY updated_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._db.close()


def review_cli() -> None:
    """``mitra-lexicon`` console entry point: list and verify pending names."""
    import argparse

    parser = argparse.ArgumentParser(prog="mitra-lexicon",
                                     description="Review model-generated Sanskrit names")
    parser.add_argument("--db", default="data/lexicon.db", help="lexicon database path")
    parser.add_argument("--verify", metavar="OBJECT_EN",
                        help="mark this object's name as human-verified")
    parser.add_argument("--name", metavar="DEVANAGARI",
                        help="corrected Devanagari name (with --verify)")
    parser.add_argument("--no-dictionary", action="store_true",
                        help="skip the Cologne lookups (faster, less context)")
    args = parser.parse_args()

    store = LexiconStore(args.db)
    if args.verify:
        store.verify(args.verify, args.name)
        print(f"verified: {args.verify}")
        return
    pending = store.pending_review()
    if not pending:
        print("no unverified entries")
        return
    # What the reviewer needs is not the coinage alone but something to
    # judge it against: what the word the model invented actually means, and
    # what the dictionaries offer for the same object (DESIGN §5).
    dictionary = None
    if not args.no_dictionary:
        try:
            from mitra.lexicon.dictionary import Dictionary
            from mitra.sanskrit import Analyzer

            analyzer = Analyzer()
            dictionary = Dictionary(analyzer=analyzer if analyzer.available else None)
        except ImportError:
            dictionary = None

    for row in pending:
        print(f"{row['object_en']:20} {row['name_devanagari']:20} "
              f"{row['name_iast'] or '-':20} ({row['updated_at']})")
        if dictionary is not None and dictionary.available:
            senses = dictionary.define(row["name_devanagari"], limit=1)
            if senses:
                print(f"    means:   {senses[0][:110]}")
            else:
                print("    means:   (not in Monier-Williams — likely coined)")
            suggested = dictionary.suggestions(row["object_en"])
            if suggested:
                print(f"    Apte:    {', '.join(suggested[:5])}")
    print(f"\n{len(pending)} entries. Verify with: "
          f"mitra-lexicon --verify OBJECT --name नाम")

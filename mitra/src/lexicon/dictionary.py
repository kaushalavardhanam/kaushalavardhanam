"""Cologne dictionary lookups: what a word means, and what word to use.

Two questions the morphology layer cannot answer. ``mitra.sanskrit`` knows
that मक्खनम् is not a Sanskrit form and that खेलानि is not on Mitra's list; it
has no idea that the word for butter is नवनीतम्. Apte's English-Sanskrit
dictionary does, and that is a word-CHOICE tool rather than a grammar one.

Used by ``mitra-lexicon``, the review CLI: every unverified name the model
coined is shown with what Monier-Williams says it means and what Apte offers
for the same English word, so a reviewer can accept or correct in one pass.

DELIBERATELY NOT WIRED INTO THE SPEAKING PATH. Substituting Apte's first
answer for the model's coinage was tried and is wrong: Apte's entries carry
idioms as well as words, so "apple" leads with तारा (from "apple of the eye")
and "teacher" with शास्. Verified lexicon rows override generation (FR-2.5);
a dictionary that is right about butter and wrong about apples is a suggestion
for a human, not an authority over the child's answer.

Absent data is normal: ``available`` is False and every lookup returns
nothing, exactly as when the phrasebook or the kosha is missing (FR-6.4).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger("mitra")

DEFAULT_PATH = "data/cdsl/cologne.db"


class Dictionary:
    def __init__(self, path: str | Path = DEFAULT_PATH,
                 analyzer=None, logger_: logging.Logger | None = None):
        self.logger = logger_ or logger
        self.analyzer = analyzer
        self.path = Path(path)
        self._db = None
        self.available = False
        if not self.path.exists():
            self.logger.info(
                "Cologne dictionary not built at %s — word-choice lookups are "
                "off (scripts/fetch_sanskrit_data.py --cologne, then "
                "scripts/build_dictionary.py)", self.path)
            return
        try:
            self._db = sqlite3.connect(str(self.path), check_same_thread=False)
            self.available = True
        except sqlite3.Error:
            self.logger.exception("could not open %s", self.path)

    # ------------------------------------------------------------ Sanskrit →

    def define(self, word: str, limit: int = 3) -> list[str]:
        """Monier-Williams senses for a Devanagari (or SLP1) word."""
        if not self.available:
            return []
        key = self.analyzer.to_slp1(word) if self.analyzer is not None else word
        # The stem is what MW is keyed on, so an inflected form needs its
        # lemma first — गृहे is not a headword, गृह is.
        keys = [key]
        if self.analyzer is not None:
            keys.extend(sorted(self.analyzer.lemmas(word)))
        for candidate in dict.fromkeys(keys):
            rows = self._db.execute(
                "SELECT meaning FROM mw WHERE key = ? LIMIT ?",
                (candidate, limit)).fetchall()
            if rows:
                return [row[0] for row in rows]
        return []

    # ------------------------------------------------------------ → Sanskrit

    def sanskrit_for(self, english: str, limit: int = 6) -> list[str]:
        """Apte's Sanskrit equivalents for an English word, best first.

        Returned in Devanagari when the morphology layer is present to
        transliterate, else as the SLP1 Apte stores.
        """
        if not self.available:
            return []
        row = None
        for key in _english_keys(english):
            row = self._db.execute(
                "SELECT sanskrit FROM ae WHERE word = ? LIMIT 1", (key,)).fetchone()
            if row:
                break
        if not row:
            return []
        words = [w for w in row[0].split(",") if w][:limit]
        if self.analyzer is None:
            return words
        from vidyut.lipi import Scheme, transliterate

        return [transliterate(w, Scheme.Slp1, Scheme.Devanagari) for w in words]

    def suggestions(self, english: str, vocabulary=None) -> list[str]:
        """Ranked Sanskrit candidates for an English word, for human review.

        Apte's own ordering, with two adjustments: forms the morphology layer
        does not recognise drop out, and anything readable as a finite verb
        sinks to the back (Apte answers "teacher" with अध्यापयति, "he
        teaches", before the agent noun). Ranked, never reduced to one — the
        list is shown to a reviewer, who decides.
        """
        # Wider than the display limit: Apte answers "teach" with the verbs
        # first and the agent nouns (अध्यापकः, शिक्षकः) several entries later,
        # and an object's NAME is what is wanted here.
        candidates = self.sanskrit_for(english, limit=12)
        if not candidates:
            return []
        if self.analyzer is None or not self.analyzer.available:
            return candidates
        # Stable partition, not a filter: a word that can be read as a finite
        # verb goes to the back but is still available. Apte answers "teacher"
        # with अध्यापयति ("he teaches") before अध्यापकः ("a teacher"), and only
        # the second is a name.
        def is_finite_verb(word: str) -> bool:
            analyses = self.analyzer.analyses(word)
            return bool(analyses) and any(a.is_verb for a in analyses)

        candidates = ([c for c in candidates if not is_finite_verb(c)]
                      + [c for c in candidates if is_finite_verb(c)])
        # One pass, in Apte's order. Scanning for a vocabulary hit first would
        # override that ordering with an accident of what the word list
        # happens to contain — asked for "ball" it picked तारा ("star", which
        # is on the list) over कन्दुकः, which is the word Apte puts first.
        known = [c for c in candidates
                 if (vocabulary is not None and vocabulary.contains(c))
                 or self.analyzer.is_attested(c)]
        return known or candidates


def _english_keys(english: str) -> list[str]:
    """Apte's headword, then the obvious reductions.

    Apte indexes "teach", not "teacher", and "book", not "books". Object
    labels arrive from vision in whatever form the model wrote them, so a
    plural or an agent noun should not silently mean "no Sanskrit word for
    this".
    """
    word = english.strip().lower()
    keys = [word]
    for suffix, replacement in (("ies", "y"), ("es", ""), ("s", ""),
                                ("er", ""), ("ing", "")):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            keys.append(word[:-len(suffix)] + replacement)
    return list(dict.fromkeys(keys))

"""The words Mitra is allowed to say (DESIGN §5, FR-3.5).

Attestation (``mitra.sanskrit.Analyzer``) answers "is this a Sanskrit word?".
It cannot answer "is this the right word", and that is where the failures
actually live: आज, घरे, खेलानि, दालः are Hindi as used, but each one is also a
real Sanskrit form of something else, so a lexicon lookup waves them through
exactly as the Devanagari script check did.

A whitelist is the only check that closes this. A blocklist grows every time
the model reaches for a new Hindi noun; the set of words a child's Sanskrit
robot needs does not grow at all. Membership is by LEMMA, so one entry covers
a word's whole paradigm: क्रीडति in the list admits क्रीडामि, क्रीडसि,
क्रीडिष्यामि.

Sources, in order of authority:

1. ``vocabulary.jsonl`` — Open Pathshala's 500-word beginner list, 28 everyday
   categories (built by scripts/build_vocabulary.py).
2. ``seed_lexicon.json`` — the human-verified object names the vision path
   already treats as authoritative (FR-2.5).
3. The phrasebook corpus, when present. The model is instructed to imitate
   those sentences, so refusing their vocabulary would reject the register we
   are actively steering it toward.
4. The fixed phrases in ``prompts.py``, which the orchestrator speaks itself.

Nothing here is a claim about what Sanskrit contains — only about what this
robot says. Words outside it are reported, not corrected, and the policy for
what to do about them lives in the validator.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger("mitra")

DEFAULT_PATH = Path(__file__).parent / "vocabulary.jsonl"

# A prefixed verb gets its own lemma in the kosha — आगच्छति is Agam, not gam,
# and प्रविशति is praviS, not viS. The word list cannot enumerate every
# prefix of every root, so a lemma is also accepted when stripping a preverb
# leaves a lemma that IS listed: प्रपठति (prapaW → paW) passes on पठति's
# entry. Longest first, so "prati" is not read as "pra" + "ti".
_UPASARGAS = ("aBisam", "samud", "samA", "prati", "pari", "anu", "apa", "aBi",
              "aDi", "ava", "vi", "nis", "nir", "dus", "dur", "sam", "upa",
              "ud", "ni", "pra", "parA", "ati", "api", "su", "A")


class Vocabulary:
    """Allowed lemmas and forms, assembled from the sources above.

    ``available`` is False when the morphology layer is missing, in which case
    every membership question answers True — an absent checker must not
    silence Mitra (FR-6.4).
    """

    def __init__(self, analyzer, path: str | Path = DEFAULT_PATH,
                 seed_path: str | Path | None = None,
                 extra_texts: tuple[str, ...] = (),
                 logger_: logging.Logger | None = None):
        self.logger = logger_ or logger
        self.analyzer = analyzer
        self.lemmas: set[str] = set()
        self.forms: set[str] = set()
        # One-syllable words that are LISTED in their own right (मा, न, किम्,
        # सः). A split word's short parts are checked against this and not
        # against the lemma set — see contains().
        self.short_forms: set[str] = set()
        self.available = bool(analyzer is not None and analyzer.available)
        if not self.available:
            return

        path = Path(path)
        entries = 0
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    entries += 1
                    self.lemmas.update(row.get("lemmas", ()))
                    for form in row.get("forms", ()):
                        self.forms.add(analyzer.canonical(form))
                    self.forms.add(analyzer.canonical(row["devanagari"]))
                    self._note_short(row["devanagari"])
        else:
            self.logger.warning("vocabulary list not found at %s", path)

        if seed_path is not None:
            self._add_seed(Path(seed_path))
        for text in extra_texts:
            self.absorb(text)

        self.logger.info("vocabulary: %d words -> %d lemmas, %d forms",
                         entries, len(self.lemmas), len(self.forms))

    # ------------------------------------------------------------ building

    def absorb(self, text: str) -> None:
        """Allow every word in a human-authored Sanskrit text."""
        if not self.available:
            return
        for word in self.analyzer.words(text):
            self._note_short(word)
            lemmas = self.analyzer.lemmas(word)
            if lemmas:
                self.lemmas.update(lemmas)
            else:
                self.forms.add(self.analyzer.canonical(word))

    def _add_seed(self, seed_path: Path) -> None:
        if not seed_path.exists():
            return
        try:
            data = json.loads(seed_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.logger.exception("could not read seed lexicon %s", seed_path)
            return
        for entry in data.get("entries", ()):
            self.absorb(entry.get("name_devanagari", ""))

    # ------------------------------------------------------------ checking

    def _note_short(self, word: str) -> None:
        if self.analyzer.syllables(word) < 2:
            self.short_forms.add(self.analyzer.canonical(word))

    def short_ok(self, word: str) -> bool:
        """True if a one-syllable fragment is a word Mitra actually knows."""
        return (not self.available
                or self.analyzer.canonical(word) in self.short_forms)

    def contains(self, word: str) -> bool:
        if not self.available:
            return True
        if self._contains_simple(word):
            return True
        # कथमस्ति is कथम् + अस्ति, and a compound is in vocabulary exactly
        # when all of its parts are — with the same fragment guard the
        # attestation check uses, and for the same reason. Without it the
        # segmenter cut गेम ("game", English in Devanagari) into गा + इम् + अ,
        # each of which is a form of some listed word, and the whole passed.
        # A one-syllable part now has to be a word IN the list, not merely a
        # form of something in it.
        parts = self.analyzer.segments(word)
        if not parts:
            return False
        return all(self._contains_simple(part)
                   and (self.analyzer.syllables(part) >= 2 or self.short_ok(part))
                   for part in parts)

    def _contains_simple(self, word: str) -> bool:
        if self.analyzer.canonical(word) in self.forms:
            return True
        # Asymmetric on purpose. The whitelist is BUILT from stem lemmas only
        # (Analyzer.lemmas defaults to basic_only), so शृणोति contributes श्रु
        # and nothing else. Membership is tested against EVERY reading, so the
        # infinitive श्रोतुम् — whose own stem reading is "Srotu" — still
        # matches through its root.
        #
        # Widening only the test side is what keeps this safe: आज reads as
        # {Aaj, AjA, Aja, aj}, and अजा ("goat") contributed only its stems
        # {ajA, aja}. The shared root अज् is never in the whitelist, so आज is
        # still rejected while श्रोतुम् is allowed.
        lemmas = self.analyzer.lemmas(word, basic_only=False)
        if lemmas & self.lemmas:
            return True
        return any(_strip_upasarga(lemma) in self.lemmas for lemma in lemmas)

    def unknown(self, text: str) -> list[str]:
        """Words in ``text`` that are outside the vocabulary.

        Duplicates are dropped but order is kept, so the reason string reads
        the way the sentence does.
        """
        if not self.available:
            return []
        seen, out = set(), []
        for word in self.analyzer.words(text):
            if word in seen or self.contains(word):
                continue
            seen.add(word)
            out.append(word)
        return out


def _strip_upasarga(lemma: str) -> str:
    """आगच्छति's lemma Agam → gam, so पठति's entry covers प्रपठति too."""
    for prefix in _UPASARGAS:
        if lemma.startswith(prefix) and len(lemma) > len(prefix) + 1:
            return lemma[len(prefix):]
    return lemma

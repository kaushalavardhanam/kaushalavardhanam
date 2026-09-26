"""Morphological lookup over the vidyut kosha (DESIGN §5).

WHY THIS EXISTS
---------------
``validator.py`` could only check *script*, and Hindi is written in the same
script: खेलानि, आज, घरे, मक्खनम् all score a Devanagari ratio of 1.00 and were
spoken to the child. The Hindi-marker stoplist that followed catches a fixed
list and nothing else — every new Hindi noun needs a new entry, and it says
nothing about करोष्यसि or कुरुमि, which are not words in any language.

This module inverts the question: instead of listing what is forbidden, look
each word up in a lexicon of every inflected Sanskrit form and reject what is
not there. The kosha also returns each form's morphology (person, number,
gender, case), which is what makes the agreement checks in ``grammar.py``
possible.

WHAT IT CANNOT DO (measured, not assumed)
-----------------------------------------
Attestation alone does not catch Hindi words that happen to be homographs of
real Sanskrit ones: आज is a genuine form (ājá), घरे parses from a root, and
खेलानि analyses as a form of खेल्. Those need the vocabulary whitelist in
``vocabulary.py``. Attestation catches the invented words; the whitelist
catches the borrowed ones. Both are needed.

NORMALIZATION
-------------
The kosha stores forms *before* external sandhi: अहम् not अहं, भवतस् not
भवतः, पुनर् not पुनः. Sanskrit as actually written uses the sandhi forms, so
a raw lookup misses them — that alone took coverage of the human-authored
phrasebook from 82.7% down to 51.6%. Every lookup therefore tries the
anusvāra and visarga variants too.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger("mitra")

DEFAULT_DATA_DIR = "data/vidyut"

# Sanskrit writes the same nasal two ways — सङ्गीतम् and संगीतम् are the same
# word — and the kosha indexes them separately: saNgItam resolves to the stem
# saMgIta, saMgItam only to a krdanta of गै. So the spelling the model happens
# to choose decided whether its own vocabulary list matched. Both directions
# are normalized: lookups try the homorganic nasal, comparisons use anusvāra.
_HOMORGANIC = {
    **{c: "N" for c in "kKgGN"}, **{c: "Y" for c in "cCjJY"},
    **{c: "R" for c in "wWqQR"}, **{c: "n" for c in "tTdDn"},
    **{c: "m" for c in "pPbBm"},
}
_CONSONANT = set("kKgGNcCjJYwWqQRtTdDnpPbBmyrlvSzsh")

# Devanagari letters and marks, minus the danda (U+0964), double danda
# (U+0965) and the digits (U+0966-096F) — none of them word characters.
_WORD = re.compile(r"[ऀ-ॣॲ-ॿ]+")


@dataclass(frozen=True)
class Analysis:
    """One possible reading of one word. A word usually has several.

    ``kind`` separates the three ways a form can be derived, and the
    distinction is load-bearing for the vocabulary check. A *krdanta* reading
    derives a noun from a verb root, and almost every word has one: आज (Hindi
    "today") and अजा ("goat") are unrelated words that nevertheless share the
    krdanta root अज्. Matching on any shared lemma therefore lets आज in
    through अजा's back door. Matching on the *basic* stem does not.
    """

    lemma: str
    is_verb: bool
    kind: str = "basic"           # verb | basic | krdanta
    purusha: str | None = None    # uttama (1st) | madhyama (2nd) | prathama (3rd)
    vacana: str | None = None     # eka | dvi | bahu
    linga: str | None = None      # puM | strI | napuMsaka
    vibhakti: str | None = None   # praTamA … saptamI, samboDanam
    is_avyaya: bool = False


class Analyzer:
    """Morphological lookup. Absent data degrades to ``available = False``.

    A missing kosha must never take the pipeline down: like the phrasebook,
    the system simply runs without this check rather than refusing to start
    (FR-6.4). Callers test ``available`` and skip the grammar checks.
    """

    def __init__(self, data_dir: str | Path = DEFAULT_DATA_DIR,
                 logger_: logging.Logger | None = None):
        self.logger = logger_ or logger
        self.data_dir = Path(data_dir)
        self._kosha = None
        self.available = False
        try:
            from vidyut.kosha import Kosha
            from vidyut.lipi import Scheme, transliterate
        except ImportError:
            self.logger.warning(
                "vidyut not installed — Sanskrit morphology checks are off. "
                "Install with: pip install 'mitra[sanskrit]'")
            return
        kosha_dir = self.data_dir / "kosha"
        if not kosha_dir.exists():
            self.logger.warning(
                "vidyut data missing at %s — Sanskrit morphology checks are "
                "off. Fetch it with: python3 scripts/fetch_sanskrit_data.py",
                kosha_dir)
            return
        try:
            self._kosha = Kosha(kosha_dir)
        except Exception:
            self.logger.exception("could not load the kosha; continuing without it")
            return
        self._transliterate = transliterate
        self._scheme = Scheme
        # The segmenter is optional: without it, sandhi-fused and compound
        # words simply read as unattested. With it they are split first.
        self._chedaka = None
        try:
            from vidyut.cheda import Chedaka

            self._chedaka = Chedaka(self.data_dir)
        except Exception:
            self.logger.warning("vidyut segmenter unavailable; compounds and "
                                "sandhi will read as unattested")
        self.available = True
        self.logger.info("sanskrit: kosha loaded from %s", kosha_dir)

    # ---------------------------------------------------------- tokenizing

    @staticmethod
    def words(text: str) -> list[str]:
        """Devanagari word tokens, danda and punctuation dropped."""
        return _WORD.findall(text or "")

    def to_slp1(self, word: str) -> str:
        return self._transliterate(word, self._scheme.Devanagari, self._scheme.Slp1)

    def _keys(self, word: str) -> tuple[str, ...]:
        """Lookup keys for one word: as written, plus its pre-sandhi forms."""
        slp = self.to_slp1(word)
        keys = [slp]
        if slp.endswith("M"):
            keys.append(slp[:-1] + "m")
        elif slp.endswith("H"):
            keys.append(slp[:-1] + "s")
            keys.append(slp[:-1] + "r")
        spelled_out = _spell_nasals(slp)
        if spelled_out != slp:
            keys.append(spelled_out)
            if spelled_out.endswith("M"):
                keys.append(spelled_out[:-1] + "m")
        return tuple(dict.fromkeys(keys))

    # ------------------------------------------------------------ analysis

    def canonical(self, word: str) -> str:
        """One stable key per word, so ``भवतः`` and ``भवतस्`` compare equal.

        Sanskrit writes the same word two ways depending on what follows it
        (अहम्/अहं, पुनर्/पुनः). Sets keyed on the raw spelling would hold both
        and match neither reliably, so everything is stored and looked up
        under the pre-sandhi form the kosha itself uses.
        """
        if not self.available:
            return word
        slp = _anusvara(self.to_slp1(word))
        if slp.endswith("M"):
            return slp[:-1] + "m"
        if slp.endswith("H"):
            return slp[:-1] + "s"
        return slp

    def analyses(self, word: str) -> list[Analysis]:
        """Every reading the kosha has for ``word``; [] if it has none."""
        if not self.available:
            return []
        return list(self._analyses_cached(word))

    @lru_cache(maxsize=4096)
    def _analyses_cached(self, word: str) -> tuple[Analysis, ...]:
        out: list[Analysis] = []
        for key in self._keys(word):
            for entry in self._kosha.get(key):
                is_verb = getattr(entry, "purusha", None) is not None
                out.append(Analysis(
                    lemma=entry.lemma,
                    is_verb=is_verb,
                    kind="verb" if is_verb else _kind(entry),
                    purusha=_name(getattr(entry, "purusha", None)),
                    vacana=_name(getattr(entry, "vacana", None)),
                    linga=_name(getattr(entry, "linga", None)),
                    vibhakti=_name(getattr(entry, "vibhakti", None)),
                    is_avyaya=bool(getattr(entry, "is_avyaya", False)),
                ))
            if out:
                break                       # first key that hits wins
        return tuple(out)

    def segments(self, word: str) -> list[str]:
        """Split a sandhi-fused or compound word into its parts.

        Sanskrit writes मा + अस्तु as मास्तु and किम् + अपि as किमपि, and the
        kosha holds single words only — so without this, ordinary correct
        sentences read as full of non-words. Measured on the phrasebook, this
        one step is the difference between 90% of human-authored sentences
        being flagged and 15%.

        Returns [] when the word does not split, INCLUDING when it is already
        a single known word. The segmenter will happily force a split on a
        genuine non-word (करोष्यसि → करोषि + असि), which is why this is only
        ever a fallback after a direct lookup fails, and why the agreement
        check has to catch what it lets through.
        """
        if not self.available or self._chedaka is None:
            return []
        return list(self._segments_cached(word))

    @lru_cache(maxsize=4096)
    def _segments_cached(self, word: str) -> tuple[str, ...]:
        try:
            tokens = self._chedaka.run(self.to_slp1(word))
        except Exception:
            return ()
        if len(tokens) < 2:
            return ()
        return tuple(
            self._transliterate(t.text, self._scheme.Slp1, self._scheme.Devanagari)
            for t in tokens)

    def parse(self, text: str) -> list[tuple[str, Analysis]]:
        """One reading per word, disambiguated in context; [] if it fails.

        The kosha returns every possible reading, and for agreement checking
        that is worse than useless: भवतः is the genitive of भवत् *and* the
        3rd-person dual of भू, so "अहं भवतः मित्रम् अस्मि" looks like a person
        clash to anything that considers all readings. The segmenter picks one
        reading per word from the sentence, which is the only way to tell a
        noun from its homographic verb.
        """
        if not self.available or self._chedaka is None:
            return []
        try:
            tokens = self._chedaka.run(self.to_slp1(text))
        except Exception:
            return []
        out = []
        for token in tokens:
            entry = token.data
            is_verb = getattr(entry, "purusha", None) is not None
            out.append((
                self._transliterate(token.text, self._scheme.Slp1,
                                    self._scheme.Devanagari),
                Analysis(
                    lemma=token.lemma or "",
                    is_verb=is_verb,
                    kind="verb" if is_verb else _kind(entry),
                    purusha=_name(getattr(entry, "purusha", None)),
                    vacana=_name(getattr(entry, "vacana", None)),
                    linga=_name(getattr(entry, "linga", None)),
                    vibhakti=_name(getattr(entry, "vibhakti", None)),
                    is_avyaya=bool(getattr(entry, "is_avyaya", False)),
                )))
        return out

    @staticmethod
    def syllables(word: str) -> int:
        """Vowel count — an akshara each. मा is one, खनम् is two."""
        count = 0
        for i, ch in enumerate(word):
            if "अ" <= ch <= "औ" or "ा" <= ch <= "ौ" or ch == "ृ":
                count += 1
            elif "क" <= ch <= "ह" and not (
                    i + 1 < len(word) and word[i + 1] in "ा-ौ्ृ"):
                count += 1        # consonant carrying its inherent vowel
        return count

    def is_attested(self, word: str, allow_part=None) -> bool:
        """True if ``word`` is a form of some Sanskrit word.

        A word that does not resolve directly is retried as a compound, and
        every part must resolve for the whole to count.

        ``allow_part`` guards the one-syllable parts, and without it this check
        loses the words it exists to catch: the segmenter will cut मक्खनम् into
        मक् + खनम् and कुरुमि into कुरुम् + इ, both halves "attested", so two
        invented words read as valid compounds. Genuine one-syllable words —
        मा in मास्तु, किम् in किमपि — are function words, which is exactly what
        the vocabulary whitelist holds, so the caller passes its membership
        test here.
        """
        if self.analyses(word):
            return True
        parts = self.segments(word)
        if not parts:
            return False
        for part in parts:
            if not self.analyses(part):
                return False
            if self.syllables(part) < 2 and not (allow_part and allow_part(part)):
                return False
        return True

        # NOT DONE, deliberately: falling back to a plain two-way split for
        # compounds the segmenter returns whole (जयनगरं, प्रवेशपत्रं). Tried
        # and measured — the kosha holds so many short forms that every
        # invented word finds a "split" too: मक्खनम्, कुरुमि, करोष्यसि, दालः
        # and even क्या all passed. A long nominal compound reading as
        # unattested is a false positive Mitra's own register rarely triggers;
        # letting मक्खनम् through is the failure this check exists to stop.

    def lemmas(self, word: str, basic_only: bool = True) -> set[str]:
        """Lemmas ``word`` could be a form of (SLP1).

        By default only the stem readings — the verb root for a finite verb,
        the underived stem for a noun. Krdanta readings are included only for
        words that have no other reading (संगीतम्, गणितम् are krdantas and
        nothing else), which is what keeps the identity of a word tied to what
        it actually is rather than to any root it can be traced back to.
        """
        analyses = self.analyses(word)
        if basic_only:
            stems = {a.lemma for a in analyses if a.kind in ("verb", "basic")}
            if stems:
                return stems
        return {a.lemma for a in analyses}

    def unattested(self, text: str, allow_part=None) -> list[str]:
        """Words in ``text`` that are not Sanskrit forms at all."""
        if not self.available:
            return []
        return [w for w in self.words(text)
                if not self.is_attested(w, allow_part)]


def _spell_nasals(slp: str) -> str:
    """संगीतम् → सङ्गीतम्: anusvāra to the nasal its neighbour implies."""
    out = list(slp)
    for i, ch in enumerate(out[:-1]):
        if ch == "M" and out[i + 1] in _HOMORGANIC:
            out[i] = _HOMORGANIC[out[i + 1]]
    return "".join(out)


def _anusvara(slp: str) -> str:
    """सङ्गीतम् → संगीतम्: any nasal before a consonant back to anusvāra."""
    out = list(slp)
    for i, ch in enumerate(out[:-1]):
        if ch in "NYRnm" and out[i + 1] in _CONSONANT:
            out[i] = "M"
    return "".join(out)


def _kind(entry) -> str:
    """'basic' or 'krdanta', from the kosha's pratipadika entry type."""
    pratipadika = getattr(entry, "pratipadika_entry", None)
    name = type(pratipadika).__name__ if pratipadika is not None else ""
    return "krdanta" if name.endswith("Krdanta") else "basic"


# vidyut names its enum members in SLP1, so the raw strings are "maDyama",
# "praTamA", "napuMsaka" — where D is dh and T is th. Lowercasing alone turns
# maDyama into "madyama", which silently matches nothing.
_SLP1_ASCII = {
    "K": "kh", "G": "gh", "C": "ch", "J": "jh", "W": "th", "Q": "dh",
    "T": "th", "D": "dh", "P": "ph", "B": "bh", "S": "sh", "z": "sh",
    "N": "n", "Y": "n", "R": "n", "w": "t", "q": "d", "M": "m", "H": "h",
    "A": "a", "I": "i", "U": "u", "E": "ai", "O": "au", "f": "ri", "x": "li",
}


def _name(value) -> str | None:
    """vidyut enum → a readable name ('Purusha.MaDyama' → 'madhyama')."""
    if value is None:
        return None
    text = str(value).rsplit(".", 1)[-1]
    return "".join(_SLP1_ASCII.get(ch, ch.lower()) for ch in text)

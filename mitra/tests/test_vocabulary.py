"""Unit tests for the MITRA vocabulary module.

These tests use a FakeAnalyzer stub so no Sanskrit data/resources are needed.
"""
import inspect
import json
import logging
import os
import sys

import pytest

try:
    from mitra import vocabulary
except ImportError:  # pragma: no cover - fallback when run from inside mitra/
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import vocabulary


class FakeAnalyzer:
    """Dict-driven stand-in for the Sanskrit analyzer.

    Every lookup falls back to a trivial default when the key is missing.
    """

    def __init__(self, available=True, canonical=None, words=None,
                 lemmas=None, segments=None, syllables=None):
        self.available = available
        self._canonical = dict(canonical or {})
        self._words = dict(words or {})
        self._lemmas = dict(lemmas or {})
        self._segments = dict(segments or {})
        self._syllables = dict(syllables or {})

    def canonical(self, text):
        return self._canonical.get(text, text)

    def words(self, text):
        return list(self._words.get(text, text.split()))

    def lemmas(self, word, basic_only=True):
        return list(self._lemmas.get(word, []))

    def segments(self, word):
        return list(self._segments.get(word, [word]))

    def syllables(self, word):
        return list(self._syllables.get(word, [word]))


DEFAULT_ROWS = [
    {"word": "rama", "meaning": "pleasing; name of a hero"},
    {"word": "gacchati", "meaning": "goes"},
    {"word": "vana", "meaning": "forest"},
]


def write_vocab_jsonl(tmp_path, rows=None, name="vocabulary.jsonl"):
    """Write a tiny vocabulary.jsonl under tmp_path and return its path."""
    rows = DEFAULT_ROWS if rows is None else rows
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def build_vocabulary(tmp_path, analyzer=None, rows=None):
    """Build a Vocabulary from a temp jsonl file using the fake analyzer."""
    path = write_vocab_jsonl(tmp_path, rows)
    if analyzer is None:
        analyzer = FakeAnalyzer()
    return vocabulary.Vocabulary(str(path), analyzer=analyzer)


@pytest.fixture
def fake_analyzer():
    return FakeAnalyzer()


@pytest.fixture
def vocab_path(tmp_path):
    return write_vocab_jsonl(tmp_path)


@pytest.fixture
def make_vocabulary(tmp_path):
    """Factory fixture: make_vocabulary(analyzer=None, rows=None)."""
    def _make(analyzer=None, rows=None):
        return build_vocabulary(tmp_path, analyzer=analyzer, rows=rows)
    return _make


def strip_upasarga(word):
    """Call _strip_upasarga whether it is a module function or a method."""
    fn = getattr(vocabulary, "_strip_upasarga", None)
    if fn is not None:
        return fn(word)
    cls = vocabulary.Vocabulary
    raw = inspect.getattr_static(cls, "_strip_upasarga")
    if isinstance(raw, (staticmethod, classmethod)):
        return getattr(cls, "_strip_upasarga")(word)
    inst = object.__new__(cls)
    return getattr(inst, "_strip_upasarga")(word)


# --- skeleton sanity checks for the test helpers ---------------------------

def test_fake_analyzer_defaults():
    a = FakeAnalyzer()
    assert a.available is True
    assert a.canonical("x") == "x"
    assert a.words("a b") == ["a", "b"]
    assert a.lemmas("x") == []
    assert a.segments("x") == ["x"]
    assert a.syllables("x") == ["x"]


def test_fake_analyzer_configured():
    a = FakeAnalyzer(available=False, canonical={"A": "a"},
                     lemmas={"ramam": ["rama"]})
    assert a.available is False
    assert a.canonical("A") == "a"
    assert a.lemmas("ramam", basic_only=False) == ["rama"]


def test_write_vocab_jsonl(tmp_path):
    path = write_vocab_jsonl(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l) for l in lines] == DEFAULT_ROWS


# --- _strip_upasarga --------------------------------------------------------

@pytest.mark.parametrize("lemma, expected", [
    ("Agam", "gam"),
    ("prapaW", "paW"),
])
def test_strip_upasarga_removes_prefix(lemma, expected):
    assert strip_upasarga(lemma) == expected


def test_strip_upasarga_longest_prefix_wins():
    # 'prati' must be stripped as a whole, not as 'pra' + 'ti...'.
    result = strip_upasarga("pratigam")
    assert result == "gam"
    assert result != "tigam"


@pytest.mark.parametrize("lemma", ["Ag", "A", "pra"])
def test_strip_upasarga_too_short_remainder_unchanged(lemma):
    # Stripping would leave <= 1 character, so the lemma is returned as-is.
    assert strip_upasarga(lemma) == lemma


# --- unavailable analyzer ---------------------------------------------------

@pytest.mark.parametrize("analyzer", [
    None,
    FakeAnalyzer(available=False),
], ids=["no-analyzer", "unavailable-analyzer"])
def test_unavailable_analyzer_is_permissive(tmp_path, analyzer):
    # Build directly so that analyzer=None is passed through unchanged.
    path = write_vocab_jsonl(tmp_path)
    vocab = vocabulary.Vocabulary(str(path), analyzer=analyzer)
    text = "zzzunknownword qqqother"
    assert vocab.contains(text) is True
    assert vocab.unknown(text) == []
    assert vocab.short_ok(text) is True


# --- vocabulary loading -----------------------------------------------------

@pytest.mark.parametrize("key, word", [
    ("lemmas", "rama"),
    ("forms", "ramam"),
    ("devanagari", "\u0930\u093e\u092e"),
])
def test_loading_accepts_each_key(tmp_path, key, word):
    rows = [{key: [word]}]
    vocab = build_vocabulary(tmp_path, analyzer=FakeAnalyzer(), rows=rows)
    assert vocab.contains(word) is True
    assert vocab.unknown(word) == []


def test_loading_rejects_word_not_in_file(tmp_path):
    rows = [{"lemmas": ["rama"]}]
    vocab = build_vocabulary(tmp_path, analyzer=FakeAnalyzer(), rows=rows)
    assert vocab.contains("zzzunknownword") is False
    assert vocab.unknown("zzzunknownword") == ["zzzunknownword"]


def test_loading_missing_file_logs_warning(tmp_path, caplog):
    missing = tmp_path / "does_not_exist.jsonl"
    assert not missing.exists()
    with caplog.at_level(logging.WARNING):
        vocab = vocabulary.Vocabulary(str(missing), analyzer=FakeAnalyzer())
    assert vocab is not None
    assert any(r.levelno >= logging.WARNING for r in caplog.records)


# --- contains() with preverb stripping --------------------------------------

def test_contains_accepts_lemma_after_preverb_stripping(tmp_path):
    # The word's lemma is 'Agam' (preverb A + gam), but only 'gam' is listed.
    analyzer = FakeAnalyzer(lemmas={"agacchati": ["Agam"]})
    rows = [{"lemmas": ["gam"]}]
    vocab = build_vocabulary(tmp_path, analyzer=analyzer, rows=rows)
    assert vocab.contains("agacchati") is True
    assert vocab.unknown("agacchati") == []


def test_contains_rejects_lemma_not_listed_even_after_stripping(tmp_path):
    # 'Aqrs' strips to 'qrs', which is not in the vocabulary either.
    analyzer = FakeAnalyzer(lemmas={"aqrsati": ["Aqrs"]})
    rows = [{"lemmas": ["gam"]}]
    vocab = build_vocabulary(tmp_path, analyzer=analyzer, rows=rows)
    assert vocab.contains("aqrsati") is False
    assert vocab.unknown("aqrsati") == ["aqrsati"]


# --- compound check and short_forms -----------------------------------------

# Multi-syllable parts must be declared explicitly: the fake analyzer treats
# every unconfigured word as a single syllable.
MULTI_SYLLABLES = {
    "rama": ["ra", "ma"],
    "vana": ["va", "na"],
    "zzzunk": ["zzz", "unk"],
}


def test_compound_accepted_when_all_parts_known(tmp_path):
    analyzer = FakeAnalyzer(
        segments={"ramavana": ["rama", "vana"]},
        lemmas={"rama": ["rama"], "vana": ["vana"]},
        syllables=MULTI_SYLLABLES,
    )
    rows = [{"lemmas": ["rama"]}, {"lemmas": ["vana"]}]
    vocab = build_vocabulary(tmp_path, analyzer=analyzer, rows=rows)
    assert vocab.contains("ramavana") is True
    assert vocab.unknown("ramavana") == []


def test_compound_rejected_when_a_part_is_unknown(tmp_path):
    analyzer = FakeAnalyzer(
        segments={"ramazzzunk": ["rama", "zzzunk"]},
        lemmas={"rama": ["rama"]},
        syllables=MULTI_SYLLABLES,
    )
    rows = [{"lemmas": ["rama"]}]
    vocab = build_vocabulary(tmp_path, analyzer=analyzer, rows=rows)
    assert vocab.contains("ramazzzunk") is False
    assert vocab.unknown("ramazzzunk") == ["ramazzzunk"]


def test_compound_one_syllable_part_accepted_if_in_short_forms(tmp_path):
    analyzer = FakeAnalyzer(
        segments={"ramaca": ["rama", "ca"]},
        lemmas={"rama": ["rama"]},
        syllables=MULTI_SYLLABLES,  # 'ca' defaults to a single syllable
    )
    rows = [{"lemmas": ["rama"]}, {"short_forms": ["ca"]}]
    vocab = build_vocabulary(tmp_path, analyzer=analyzer, rows=rows)
    assert vocab.contains("ramaca") is True
    assert vocab.unknown("ramaca") == []


def test_compound_one_syllable_part_rejected_if_only_form_of_lemma(tmp_path):
    # 'sa' is a form of the listed lemma 'tad', but it is not in short_forms,
    # so as a one-syllable compound part it must be rejected.
    analyzer = FakeAnalyzer(
        segments={"ramasa": ["rama", "sa"]},
        lemmas={"rama": ["rama"], "sa": ["tad"]},
        syllables=MULTI_SYLLABLES,
    )
    rows = [{"lemmas": ["rama"]}, {"lemmas": ["tad"]}]
    vocab = build_vocabulary(tmp_path, analyzer=analyzer, rows=rows)
    assert vocab.contains("ramasa") is False
    assert vocab.unknown("ramasa") == ["ramasa"]


# --- unknown(): dedup and ordering ------------------------------------------

def test_unknown_dedups_and_keeps_first_appearance_order(tmp_path):
    rows = [{"lemmas": ["rama"]}, {"lemmas": ["vana"]}]
    vocab = build_vocabulary(tmp_path, analyzer=FakeAnalyzer(), rows=rows)
    text = "zzzb rama zzza zzzb vana zzza zzzc rama"
    # Each unknown word appears once, ordered by first appearance;
    # known words (rama, vana) are excluded.
    assert vocab.unknown(text) == ["zzzb", "zzza", "zzzc"]


def test_unknown_all_known_repeated_words_returns_empty(tmp_path):
    rows = [{"lemmas": ["rama"]}]
    vocab = build_vocabulary(tmp_path, analyzer=FakeAnalyzer(), rows=rows)
    assert vocab.unknown("rama rama rama") == []


# --- absorb() ---------------------------------------------------------------

def test_absorb_adds_lemmas_when_analyzer_finds_them(tmp_path):
    analyzer = FakeAnalyzer(lemmas={"ramam": ["rama"]})
    vocab = build_vocabulary(tmp_path, analyzer=analyzer, rows=[])
    assert vocab.contains("ramam") is False
    vocab.absorb("ramam")
    assert vocab.contains("ramam") is True
    assert vocab.unknown("ramam") == []


def test_absorb_falls_back_to_surface_form_without_lemmas(tmp_path):
    # The analyzer knows nothing about this word, so absorb() must fall back
    # to the alternative behaviour (recording the word itself).
    vocab = build_vocabulary(tmp_path, analyzer=FakeAnalyzer(), rows=[])
    assert vocab.contains("zzzunknownword") is False
    vocab.absorb("zzzunknownword")
    assert vocab.contains("zzzunknownword") is True
    assert vocab.unknown("zzzunknownword") == []


def test_absorb_multiple_words_mixed_lemma_and_fallback(tmp_path):
    analyzer = FakeAnalyzer(lemmas={"ramam": ["rama"]})
    vocab = build_vocabulary(tmp_path, analyzer=analyzer, rows=[])
    text = "ramam zzzunknownword"
    assert vocab.unknown(text) == ["ramam", "zzzunknownword"]
    vocab.absorb(text)
    assert vocab.contains("ramam") is True
    assert vocab.contains("zzzunknownword") is True
    assert vocab.unknown(text) == []


def test_absorb_does_not_affect_other_words(tmp_path):
    analyzer = FakeAnalyzer(lemmas={"ramam": ["rama"]})
    vocab = build_vocabulary(tmp_path, analyzer=analyzer, rows=[])
    vocab.absorb("ramam")
    assert vocab.contains("qqqother") is False
    assert vocab.unknown("qqqother") == ["qqqother"]

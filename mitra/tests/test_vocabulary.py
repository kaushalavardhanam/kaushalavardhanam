"""Unit tests for the MITRA vocabulary module.

These tests use a FakeAnalyzer stub so no Sanskrit data/resources are needed.
"""
import json
import logging

import pytest

from mitra.lexicon import vocabulary


class FakeAnalyzer:
    """Dict-driven stand-in for the Sanskrit analyzer.

    Every lookup falls back to a trivial default when the key is missing.
    ``syllables`` maps word -> int and defaults to 1 (a short word).
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
        return set(self._lemmas.get(word, ()))

    def segments(self, word):
        return list(self._segments.get(word, [word]))

    def syllables(self, word):
        return self._syllables.get(word, 1)


def write_vocab_jsonl(tmp_path, rows, name="vocabulary.jsonl"):
    """Write a tiny vocabulary.jsonl under tmp_path and return its path."""
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def build_vocabulary(tmp_path, analyzer, rows, **kwargs):
    path = write_vocab_jsonl(tmp_path, rows)
    return vocabulary.Vocabulary(analyzer, path=str(path), **kwargs)


# --- helper sanity checks ---------------------------------------------------

def test_fake_analyzer_defaults():
    a = FakeAnalyzer()
    assert a.available is True
    assert a.canonical("x") == "x"
    assert a.words("a b") == ["a", "b"]
    assert a.lemmas("x") == set()
    assert a.segments("x") == ["x"]
    assert a.syllables("x") == 1


def test_fake_analyzer_configured():
    a = FakeAnalyzer(available=False, canonical={"A": "a"},
                     lemmas={"ramam": {"rama"}}, syllables={"rama": 2})
    assert a.available is False
    assert a.canonical("A") == "a"
    assert a.lemmas("ramam", basic_only=False) == {"rama"}
    assert a.syllables("rama") == 2


def test_write_vocab_jsonl(tmp_path):
    rows = [{"lemmas": ["rama"], "forms": ["ramam"], "devanagari": "rAma"}]
    path = write_vocab_jsonl(tmp_path, rows)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l) for l in lines] == rows


# --- _strip_upasarga --------------------------------------------------------

@pytest.mark.parametrize("lemma, expected", [
    ("Agam", "gam"),
    ("prapaW", "paW"),
])
def test_strip_upasarga_removes_prefix(lemma, expected):
    assert vocabulary._strip_upasarga(lemma) == expected


def test_strip_upasarga_longest_prefix_wins():
    # 'prati' must be stripped as a whole, not as 'pra' + 'ti...'.
    result = vocabulary._strip_upasarga("pratigam")
    assert result == "gam"
    assert result != "tigam"


@pytest.mark.parametrize("lemma", ["Ag", "A", "pra", "viA", "Ax"])
def test_strip_upasarga_too_short_remainder_unchanged(lemma):
    # Stripping would leave <= 1 character, so the lemma is returned as-is.
    assert vocabulary._strip_upasarga(lemma) == lemma


# --- unavailable analyzer ---------------------------------------------------

@pytest.mark.parametrize("analyzer", [
    None,
    FakeAnalyzer(available=False),
], ids=["no-analyzer", "unavailable-analyzer"])
def test_unavailable_analyzer_is_permissive(tmp_path, analyzer):
    vocab = vocabulary.Vocabulary(analyzer, path=str(tmp_path / "none.jsonl"))
    text = "zzzunknownword qqqother"
    assert vocab.contains(text) is True
    assert vocab.unknown(text) == []
    assert vocab.short_ok(text) is True


# --- vocabulary loading -----------------------------------------------------

def test_loading_accepts_lemmas(tmp_path):
    analyzer = FakeAnalyzer(lemmas={"ramasya": {"rama"}})
    rows = [{"lemmas": ["rama"], "devanagari": "other"}]
    vocab = build_vocabulary(tmp_path, analyzer, rows)
    assert vocab.contains("ramasya") is True
    assert vocab.unknown("ramasya") == []


def test_loading_accepts_forms(tmp_path):
    rows = [{"forms": ["ramam"], "devanagari": "other"}]
    vocab = build_vocabulary(tmp_path, FakeAnalyzer(), rows)
    assert vocab.contains("ramam") is True
    assert vocab.unknown("ramam") == []


def test_loading_accepts_devanagari(tmp_path):
    rows = [{"devanagari": "rAma"}]
    vocab = build_vocabulary(tmp_path, FakeAnalyzer(), rows)
    assert vocab.contains("rAma") is True
    assert vocab.unknown("rAma") == []


def test_loading_canonicalises_forms_and_devanagari(tmp_path):
    analyzer = FakeAnalyzer(canonical={"Ramam": "ramam", "RAma": "rAma"})
    rows = [{"forms": ["Ramam"], "devanagari": "RAma"}]
    vocab = build_vocabulary(tmp_path, analyzer, rows)
    assert {"ramam", "rAma"} <= vocab.forms


def test_loading_rejects_word_not_in_file(tmp_path):
    rows = [{"lemmas": ["rama"], "devanagari": "rAma"}]
    vocab = build_vocabulary(tmp_path, FakeAnalyzer(), rows)
    assert vocab.contains("zzzunknownword") is False
    assert vocab.unknown("zzzunknownword") == ["zzzunknownword"]


def test_loading_missing_file_logs_warning(tmp_path, caplog):
    missing = tmp_path / "does_not_exist.jsonl"
    assert not missing.exists()
    with caplog.at_level(logging.WARNING, logger="mitra"):
        vocab = vocabulary.Vocabulary(FakeAnalyzer(), path=str(missing))
    assert vocab.lemmas == set() and vocab.forms == set()
    assert any(r.levelno == logging.WARNING
               and "vocabulary list not found" in r.getMessage()
               for r in caplog.records)


# --- contains() with preverb stripping --------------------------------------

def test_contains_accepts_lemma_after_preverb_stripping(tmp_path):
    # The word's lemma is 'Agam' (preverb A + gam), but only 'gam' is listed.
    analyzer = FakeAnalyzer(lemmas={"agacchati": {"Agam"}})
    rows = [{"lemmas": ["gam"], "devanagari": "gacchati"}]
    vocab = build_vocabulary(tmp_path, analyzer, rows)
    assert vocab.contains("agacchati") is True
    assert vocab.unknown("agacchati") == []


def test_contains_rejects_lemma_not_listed_even_after_stripping(tmp_path):
    # 'Aqrs' strips to 'qrs', which is not in the vocabulary either.
    analyzer = FakeAnalyzer(lemmas={"aqrsati": {"Aqrs"}})
    rows = [{"lemmas": ["gam"], "devanagari": "gacchati"}]
    vocab = build_vocabulary(tmp_path, analyzer, rows)
    assert vocab.contains("aqrsati") is False
    assert vocab.unknown("aqrsati") == ["aqrsati"]


# --- compound check and short_forms -----------------------------------------

def test_compound_accepted_when_all_parts_known(tmp_path):
    analyzer = FakeAnalyzer(
        segments={"ramavana": ["rama", "vana"]},
        lemmas={"rama": {"rama"}, "vana": {"vana"}},
        syllables={"rama": 2, "vana": 2},
    )
    rows = [{"lemmas": ["rama"], "devanagari": "x1"},
            {"lemmas": ["vana"], "devanagari": "x2"}]
    vocab = build_vocabulary(tmp_path, analyzer, rows)
    assert vocab._contains_simple("ramavana") is False
    assert vocab.contains("ramavana") is True
    assert vocab.unknown("ramavana") == []


def test_compound_rejected_when_a_part_is_unknown(tmp_path):
    analyzer = FakeAnalyzer(
        segments={"ramazzzunk": ["rama", "zzzunk"]},
        lemmas={"rama": {"rama"}},
        syllables={"rama": 2, "zzzunk": 2},
    )
    rows = [{"lemmas": ["rama"], "devanagari": "x1"}]
    vocab = build_vocabulary(tmp_path, analyzer, rows)
    assert vocab.contains("ramazzzunk") is False
    assert vocab.unknown("ramazzzunk") == ["ramazzzunk"]


def test_compound_one_syllable_part_accepted_if_in_short_forms(tmp_path):
    analyzer = FakeAnalyzer(
        segments={"ramaca": ["rama", "ca"]},
        lemmas={"rama": {"rama"}},
        syllables={"rama": 2},  # 'ca' defaults to one syllable
    )
    rows = [{"lemmas": ["rama"], "devanagari": "x1"},
            {"devanagari": "ca"}]
    vocab = build_vocabulary(tmp_path, analyzer, rows)
    assert "ca" in vocab.short_forms
    assert vocab.contains("ramaca") is True
    assert vocab.unknown("ramaca") == []


def test_compound_one_syllable_part_rejected_if_only_form_of_lemma(tmp_path):
    # 'sa' is a form of the listed lemma 'tad', but it is not in short_forms,
    # so as a one-syllable compound part it must be rejected.
    analyzer = FakeAnalyzer(
        segments={"ramasa": ["rama", "sa"]},
        lemmas={"rama": {"rama"}, "sa": {"tad"}},
        syllables={"rama": 2, "tada": 2},
    )
    rows = [{"lemmas": ["rama"], "devanagari": "x1"},
            {"lemmas": ["tad"], "devanagari": "tada"}]
    vocab = build_vocabulary(tmp_path, analyzer, rows)
    assert vocab._contains_simple("sa") is True
    assert "sa" not in vocab.short_forms
    assert vocab.contains("ramasa") is False
    assert vocab.unknown("ramasa") == ["ramasa"]


# --- unknown(): dedup and ordering ------------------------------------------

def test_unknown_dedups_and_keeps_first_appearance_order(tmp_path):
    analyzer = FakeAnalyzer(lemmas={"rama": {"rama"}})
    rows = [{"lemmas": ["rama"], "devanagari": "x1"}]
    vocab = build_vocabulary(tmp_path, analyzer, rows)
    assert vocab.unknown("zzzb rama zzza zzzb zzzc") == ["zzzb", "zzza", "zzzc"]


# --- absorb() ---------------------------------------------------------------

def test_absorb_adds_lemmas_when_analyzer_finds_them(tmp_path):
    analyzer = FakeAnalyzer(lemmas={"ramam": {"rama"}})
    vocab = build_vocabulary(tmp_path, analyzer, [])
    assert vocab.contains("ramam") is False
    vocab.absorb("ramam")
    assert "rama" in vocab.lemmas
    assert "ramam" not in vocab.forms
    assert vocab.contains("ramam") is True


def test_absorb_falls_back_to_canonical_form_without_lemmas(tmp_path):
    analyzer = FakeAnalyzer(canonical={"Ramam": "ramam"})
    vocab = build_vocabulary(tmp_path, analyzer, [])
    assert vocab.contains("Ramam") is False
    vocab.absorb("Ramam")
    assert "ramam" in vocab.forms
    assert vocab.lemmas == set()
    assert vocab.contains("Ramam") is True


def test_absorb_via_extra_texts(tmp_path):
    analyzer = FakeAnalyzer(lemmas={"ramam": {"rama"}})
    vocab = build_vocabulary(tmp_path, analyzer, [],
                             extra_texts=("ramam zzzword",))
    assert "rama" in vocab.lemmas
    assert "zzzword" in vocab.forms

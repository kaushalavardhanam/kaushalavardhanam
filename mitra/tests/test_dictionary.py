"""Cologne dictionary lookups (DESIGN §5).

Skipped when the index is absent, which is also how the runtime behaves —
these lookups are a reviewer's aid, never a prerequisite for speaking.
"""

from __future__ import annotations

import pytest

from mitra.lexicon.dictionary import Dictionary
from mitra.sanskrit import Analyzer


@pytest.fixture(scope="module")
def dictionary():
    analyzer = Analyzer()
    dictionary = Dictionary(analyzer=analyzer if analyzer.available else None)
    if not dictionary.available:
        pytest.skip("Cologne index absent — run scripts/build_dictionary.py")
    return dictionary


def test_english_to_sanskrit_finds_the_word_the_model_missed(dictionary):
    """मक्खनम् (Hindi) was spoken for "butter"; Apte has नवनीत."""
    suggestions = dictionary.suggestions("butter")
    assert any("नवनीत" in s for s in suggestions)


def test_coined_words_have_no_entry(dictionary):
    """The review CLI leans on this: no MW sense means the model invented it."""
    assert dictionary.define("मक्खनम्") == []
    assert dictionary.define("नवनीतम्")


def test_inflected_forms_resolve_through_their_lemma(dictionary):
    """गृहे is not a headword; गृह is."""
    assert dictionary.define("गृहे")


def test_plurals_and_agent_nouns_fall_back_to_the_headword(dictionary):
    """Apte indexes "book", not "books"; vision labels arrive either way."""
    assert dictionary.suggestions("books")


def test_finite_verbs_rank_below_names(dictionary):
    """Apte answers "teacher" with अध्यापयति ("he teaches") before the noun."""
    suggestions = dictionary.suggestions("teacher")
    if "अध्यापयति" in suggestions:
        assert suggestions.index("अध्यापयति") > 0


def test_missing_index_is_not_an_error():
    dictionary = Dictionary("/nonexistent/cologne.db")
    assert not dictionary.available
    assert dictionary.define("गृहम्") == []
    assert dictionary.suggestions("butter") == []

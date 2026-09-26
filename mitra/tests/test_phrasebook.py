"""Phrasebook retrieval (DESIGN §4).

The real corpus is licensed and gitignored, so these build a tiny synthetic
one with the same shape: question rows, their answer rows following, and
unrelated rows to be discriminated against.
"""

from __future__ import annotations

import json

import pytest

from mitra.lexicon.phrasebook import Phrasebook, _content, _fold, _trigrams

ROWS = [
    {"id": "01-0001", "chapter": "greet", "sanskrit": "नमस्ते ।",
     "iast": "namaste |", "english": "Good evening."},
    {"id": "02-0001", "chapter": "meet", "sanskrit": "भवतः नाम किं ?",
     "iast": "bhavataḥ nāma kiṃ ?", "english": "What is your name? (masc.)"},
    {"id": "02-0002", "chapter": "meet", "sanskrit": "भवत्याः नाम किम्?",
     "iast": "bhavatyāḥ nāma kim?", "english": "What is your name? (fem.)"},
    {"id": "02-0003", "chapter": "meet", "sanskrit": "मम नाम रामः ।",
     "iast": "mama nāma rāmaḥ |", "english": "My name is Rama"},
    {"id": "02-0004", "chapter": "meet", "sanskrit": "भवतः वेतनश्रेणी का ?",
     "iast": "bhavataḥ vetanaśreṇī kā ?", "english": "What is your scale of pay ?"},
    {"id": "02-0005", "chapter": "meet", "sanskrit": "मम वेतनं अल्पम् ।",
     "iast": "mama vetanaṃ alpam |", "english": "My pay is small"},
    {"id": "03-0001", "chapter": "misc", "sanskrit": "अस्य रुचिं पश्यतु ।",
     "iast": "asya ruciṃ paśyatu |", "english": "Taste this, please."},
]


@pytest.fixture
def pb(tmp_path):
    path = tmp_path / "phrasebook.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ROWS),
        encoding="utf-8")
    return Phrasebook(path)


def _ids(rows):
    return [r["id"] for r in rows]


def test_missing_corpus_runs_ungrounded(tmp_path):
    book = Phrasebook(tmp_path / "absent.jsonl")
    assert book.count() == 0
    assert book.similar("anything") == []


def test_loads_rows(pb):
    assert pb.count() == len(ROWS)


def test_question_resolves_to_its_answer(pb):
    """The echo failure this replaced: asking a question returned that same
    question, so the model asked it back instead of answering."""
    got = _ids(pb.similar("What is your name?", k=3))
    assert "02-0003" in got, got


def test_answer_block_skips_question_variants(pb):
    """02-0001 (masc.) is followed by 02-0002 (fem.) before the answer at
    02-0003 — the masculine form must still find it."""
    idx = {r["id"]: i for i, r in enumerate(pb._rows)}
    answers = pb._answers[idx["02-0001"]]
    assert [pb._rows[j]["id"] for j in answers] == ["02-0003"]


def test_stopwords_do_not_carry_the_match(pb):
    """"What is your name?" and "What is your scale of pay?" share every
    function word; only the topical words may decide."""
    got = _ids(pb.similar("What is your name?", k=3))
    assert "02-0005" not in got, got


def test_romanised_sanskrit_matches_iast(pb):
    """Whisper romanises without diacritics and runs words together."""
    for query in ("bhavatah nama kim", "Bhafatah naamakim.", "Bhavatah Namakim"):
        assert "02-0003" in _ids(pb.similar(query, k=3)), query


def test_unrelated_query_attaches_nothing(pb):
    assert pb.similar("quantum entanglement chromatography", k=3) == []


def test_respects_k(pb):
    assert len(pb.similar("What is your name?", k=1)) == 1


def test_by_chapter(pb):
    assert _ids(pb.by_chapter("greet")) == ["01-0001"]


def test_fold_collapses_doubled_vowels():
    assert _fold("Bhavaan naama keem") == _fold("bhavān nāma kim")


def test_fold_strips_danda_pipe():
    assert _fold("dhanyavādaḥ |") == "dhanyavadah"


def test_content_drops_function_words():
    assert _content("what is your name") == "name"


def test_content_keeps_all_stopword_string():
    """A gloss that is nothing but function words must not fold to empty."""
    assert _content("what is it") == "what is it"


def test_trigrams_of_empty_string():
    assert _trigrams("") == set()


def test_user_own_line_is_not_handed_back(pb):
    """The echo guard: saying a corpus line verbatim must return what follows
    it, not the line itself — otherwise the model repeats the user."""
    got = _ids(pb.similar("mama nama ramah", k=3))
    assert "02-0003" not in got, got


def test_weak_statement_match_still_returns_the_row(pb):
    """Only a near-identical statement counts as an echo; a loose match is
    ordinary reference material and must survive."""
    got = _ids(pb.similar("Tell me about your pay", k=3))
    assert "02-0005" in got, got

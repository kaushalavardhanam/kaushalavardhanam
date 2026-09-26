"""Morphology checks against known-good and known-bad Sanskrit (DESIGN §5).

The good cases are the prompt's own few-shot examples and the fixed phrases
the orchestrator speaks — hand-verified Sanskrit in exactly Mitra's register.
Every one of them MUST pass: a false rejection costs a retry and can end in
the safe fallback, i.e. the child asks a question and Mitra says "sorry".

The bad cases are verbatim from logs/turns.jsonl, one per failure mode we
actually shipped to a child.

Skipped when the vidyut data is absent, which is also how the runtime behaves.
"""

from __future__ import annotations

import pytest

from mitra.lexicon.vocabulary import Vocabulary
from mitra.sanskrit import Analyzer, grammar

GOOD = [
    "मम नाम मित्रम्।",                    # prompt few-shot
    "अहं कुशली अस्मि।",
    "अहं भवतः मित्रम् अस्मि।",
    "एतत् सेवफलम् अस्ति।",
    "सूर्यः आकाशे भाति।",
    "अहं पुस्तकं पठामि।",
    "अद्य अहं पठिष्यामि।",
    "मम प्रियं भोजनं नवनीतम् अस्ति।",
    "मह्यं गणितं रोचते।",
    "अहं क्रीडामि।",
    "अहं संगीतं श्रोतुम् इच्छामि।",
    "पुनः मिलामः।",                       # orchestrator fixed phrases
    "क्षम्यताम्, पुनः वदतु।",
    "पुनः दर्शयतु।",
    "क्षम्यताम्, अहं न अवगच्छामि।",
]

BAD = [
    ("अहं खेलानि करोमि।", "vocabulary"),          # Hindi खेल, Sanskrit ending
    ("अहं आज खेलानि करिष्यामि।", "vocabulary"),   # आज is Hindi for अद्य
    ("अहं घरे अस्मि।", "vocabulary"),             # घर is Hindi for गृह
    ("अहं मक्खनम् प्रियं अस्मि।", "unattested"),  # मक्खन is Hindi for नवनीत
    ("अहं खेलानि कुरुमि।", "unattested"),         # कुरुमि is not a word
    ("भवान् किं करोष्यसि?", "agreement"),          # nor is करोष्यसि
    ("भवान् किं पठसि?", "agreement"),              # भवान् is 3rd person
    ("किं त्वम् अस्मि?", "agreement"),             # त्वम् is 2nd person
]


@pytest.fixture(scope="module")
def checker():
    analyzer = Analyzer()
    if not analyzer.available:
        pytest.skip("vidyut data absent — run scripts/fetch_sanskrit_data.py")
    from pathlib import Path

    seed = Path(__file__).resolve().parents[1] / "src" / "lexicon" / "seed_lexicon.json"
    return grammar.Checker(analyzer, Vocabulary(analyzer, seed_path=seed))


@pytest.mark.parametrize("sentence", GOOD)
def test_correct_sanskrit_is_not_rejected(checker, sentence):
    findings = checker(sentence)
    assert not findings, f"false positive: {checker.reason(findings)}"


@pytest.mark.parametrize("sentence,expected", BAD)
def test_logged_failures_are_caught(checker, sentence, expected):
    checks = {finding.check for finding in checker(sentence)}
    assert checks, f"missed: {sentence}"
    assert expected in checks, f"caught by {checks}, expected {expected}"


def test_hindi_homographs_do_not_enter_through_a_shared_root(checker):
    """आज ("today", Hindi) and अजा ("goat") share the root अज्.

    Matching on any shared lemma would admit आज on अजा's entry, which is the
    reason the whitelist is built from stem readings only.
    """
    vocabulary = checker.vocabulary
    assert vocabulary.contains("अजा")
    assert not vocabulary.contains("आज")


def test_anusvara_spelling_does_not_decide_membership(checker):
    """सङ्गीतम् and संगीतम् are the same word; the kosha indexes them apart."""
    assert checker.vocabulary.contains("सङ्गीतम्")
    assert checker.vocabulary.contains("संगीतं")


def test_a_word_the_list_lacks_is_reached_through_its_root(checker):
    """पठति is listed; प्रपठति is not, but is the same verb with a preverb."""
    assert checker.vocabulary.contains("प्रपठति")


def test_unavailable_analyzer_checks_nothing():
    analyzer = Analyzer("/nonexistent/path")
    assert not analyzer.available
    assert grammar.check("अहं खेलानि करोमि।", analyzer) == []
    assert Vocabulary(analyzer).contains("मक्खनम्")   # no data, no opinions

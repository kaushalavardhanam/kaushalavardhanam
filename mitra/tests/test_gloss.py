"""English gloss of spoken Sanskrit for the debug log (FR-7.2)."""

from __future__ import annotations

import pytest

from mitra.agent import prompts
from mitra.gloss import Glosser


class FakeTranslator:
    def __init__(self, replies=("I play.",)):
        self.replies = list(replies)
        self.calls: list[str] = []
        self.resets = 0

    def converse(self, message: str) -> str:
        self.calls.append(message)
        return self.replies.pop(0) if self.replies else "..."

    def reset(self) -> None:
        self.resets += 1


def make(replies=("I play.",)):
    translator = FakeTranslator(replies)
    return Glosser(lambda: translator), translator


def test_translates_sanskrit_and_clears_history():
    glosser, translator = make(["I play."])
    assert glosser.gloss("अहं क्रीडां करोमि।") == "I play."
    assert translator.calls == ["अहं क्रीडां करोमि।"]
    assert translator.resets == 1          # each gloss sees one sentence only


def test_repeated_line_is_cached():
    glosser, translator = make(["I play."])
    glosser.gloss("अहं क्रीडां करोमि।")
    assert glosser.gloss("अहं क्रीडां करोमि।") == "I play."
    assert len(translator.calls) == 1


def test_fixed_phrases_need_no_model_call():
    glosser, translator = make()
    assert glosser.gloss(prompts.GREETING) == "Hello."
    assert glosser.gloss(prompts.FAREWELL) == "See you again."
    assert translator.calls == []


def test_english_reply_is_not_glossed():
    """An [explain_in_english] turn is already English — nothing to translate."""
    glosser, translator = make()
    assert glosser.gloss("It means 'my name is Mitra'.") is None
    assert translator.calls == []


def test_multiline_output_is_reduced_to_the_translation():
    glosser, _ = make(['"I play."\nLiteral: play I do.'])
    assert glosser.gloss("अहं क्रीडां करोमि।") == "I play."


def test_lazy_agent_is_never_built_when_nothing_needs_it():
    def explode():
        raise AssertionError("translator built unnecessarily")

    glosser = Glosser(explode)
    assert glosser.gloss(prompts.GREETING) == "Hello."


def test_failure_disables_glossing_instead_of_raising():
    calls = []

    def factory():
        calls.append(1)
        raise RuntimeError("ollama is down")

    glosser = Glosser(factory)
    assert glosser.gloss("अहं क्रीडां करोमि।") is None
    assert glosser.gloss("अहं पठामि।") is None
    assert len(calls) == 1                 # not retried on every line spoken


@pytest.mark.parametrize("junk", ["", "   ", None])
def test_empty_text_is_ignored(junk):
    glosser, translator = make()
    assert glosser.gloss(junk) is None
    assert translator.calls == []

"""Debug-only English gloss of what Mitra says (FR-7.2).

The console log shows the Devanagari the robot is about to speak, which is
exactly what an operator who does not read Sanskrit cannot check. This
translates each spoken line back into English for the log:

    INFO mitra: speak: मह्यं गणितं रोचते।
    INFO mitra: speak (en): Mathematics is pleasing to me.

Deliberately a SECOND, history-free model call rather than asking the agent
for its own translation, and deliberately not a phrasebook lookup:

* Asking the agent to append an English gloss would put English in the
  conversation history it imitates, and the one hard rule is Sanskrit-only
  output (prompts §1). The gloss must not be able to leak into speech.
* A nearest-phrasebook-row gloss would report the corpus sentence the model
  was steered by, not the sentence it actually produced — so a mangled reply
  would read as correct English. That defeats the only reason to log this.

The translator therefore renders the reply literally, warts included, and
runs off the speaking path (see Orchestrator._speak): the call happens after
playback has started, so it costs log latency, never speech latency.
"""

from __future__ import annotations

import logging
import re

from mitra.agent import prompts

GLOSS_SYSTEM_PROMPT = """\
You are a Sanskrit-to-English translation tool. The user sends one Sanskrit \
sentence; you reply with its English translation and nothing else.

Rules:
1. Output ONE line of plain English. No transliteration, no quotes, no notes, \
no explanation, no restating the Sanskrit.
2. Translate LITERALLY what the sentence actually says, including its errors. \
If it is ungrammatical, nonsensical, or says something odd, render that \
faithfully — never repair it into the sentence you think was intended.
3. If a word is not Sanskrit (e.g. Hindi), translate it and mark it, like: \
today [Hindi].

Examples:
मम नाम मित्रम्। -> My name is Mitra.
अहं क्रीडां करोमि। -> I play.
अहं गणितं प्रियम् अस्मि। -> I am dear mathematics.
अहं आज पठामि। -> I read today [आज is Hindi].
"""

# Fixed phrases the orchestrator speaks itself (prompts.py). Glossing them
# through the model would spend a call per turn on strings whose English is
# already known — and known better than an 8B model would render it.
FIXED_GLOSSES = {
    prompts.GREETING: "Hello.",
    prompts.FAREWELL: "See you again.",
    prompts.APOLOGY_RETRY: "Sorry, please say that again.",
    prompts.APOLOGY_SHOW_AGAIN: "Please show me again.",
    prompts.SAFE_FALLBACK: "Sorry, I do not understand.",
}

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# One session speaks a bounded number of distinct lines, but nothing enforces
# that — cap the cache so a long run cannot grow it without limit.
_MAX_CACHE = 256


class Glosser:
    """Translates spoken Sanskrit to English for the debug log.

    ``agent_factory`` is called lazily on first use and must return an object
    with ``converse(str) -> str`` and ``reset()`` — a plain MitraAgent with no
    tools. It is lazy so that enabling the gloss costs nothing until Mitra
    actually says something, and reset between calls so each translation sees
    one sentence and no conversation.
    """

    def __init__(self, agent_factory, logger: logging.Logger | None = None):
        self._agent_factory = agent_factory
        self._agent = None
        self.logger = logger or logging.getLogger("mitra")
        self._cache: dict[str, str] = dict(FIXED_GLOSSES)
        self._enabled = True

    def gloss(self, text: str) -> str | None:
        """English for ``text``, or None if unavailable or not needed."""
        text = (text or "").strip()
        if not text or not _DEVANAGARI.search(text):
            # Nothing to translate: an [explain_in_english] turn already
            # answered in English, and the log line carries it verbatim.
            return None
        if text in self._cache:
            return self._cache[text]
        if not self._enabled:
            return None
        try:
            if self._agent is None:
                self._agent = self._agent_factory()
            raw = self._agent.converse(text)
            self._agent.reset()
        except Exception:
            # A gloss is an operator convenience; it must never take a turn
            # down. One failure disables it for the run rather than paying
            # the same failure on every line spoken afterwards.
            self.logger.exception("English gloss failed — disabling for this run")
            self._enabled = False
            return None
        english = _first_line(raw)
        if not english:
            return None
        if len(self._cache) < _MAX_CACHE:
            self._cache[text] = english
        return english


def _first_line(raw: str) -> str:
    """First non-empty line of the model's output, unquoted and capped.

    Instructed to answer with one line, the model still sometimes adds a
    second ("Literal: ..."). The log wants the translation, not the essay.
    """
    for line in str(raw or "").splitlines():
        line = line.strip().strip('"').strip()
        if line:
            return line[:300]
    return ""

"""Whisper hallucination / no-speech filters.

Short or quiet simulator captures often decode as YouTube-style leftovers
("Thanks for watching") or as a word repeated until the window fills.
Those are ASR failures, not LLM failures.
"""

from __future__ import annotations

import re

# Phrases Whisper commonly invents on silence, music, or very short clips.
_PHRASES = (
    "thank you for watching",
    "thanks for watching",
    "thank you for listening",
    "please subscribe",
    "like and subscribe",
    "subscribe to",
    "see you next time",
    "don't forget to subscribe",
    "thanks for listening",
    "you",
    "thank you",
    "thanks",
    "bye",
    "the end",
    "music",
    "applause",
    "[music]",
    "(music)",
    "subtitles by",
    "transcript by",
)

_REPEAT = re.compile(r"^(.{2,20}?)(\s+\1){3,}$", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-zÀ-ɏऀ-ॿಕ-೯]+")


def is_hallucination(text: str, *, duration_s: float | None = None) -> bool:
    cleaned = " ".join((text or "").strip().lower().split())
    if not cleaned:
        return False
    if cleaned in _PHRASES:
        return True
    if any(cleaned.startswith(p) and len(cleaned) < len(p) + 16 for p in _PHRASES[:10]):
        return True
    if _REPEAT.match(cleaned):
        return True
    words = _WORD.findall(cleaned)
    if duration_s is not None and duration_s < 2.0 and len(words) > 16:
        return True
    if len(words) >= 6 and len(set(w.lower() for w in words)) == 1:
        return True
    return False


def looks_unusable(text: str) -> bool:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return True
    if is_hallucination(cleaned):
        return True
    if not _WORD.search(cleaned):
        return True
    return False

"""Per-utterance language detection for en/sa (FR-3.1).

Script-range heuristic over the ASR transcript: Devanagari block → sa, Latin
letters → en; majority wins. The ASR engine's own language tag is used only
as a tie-break hint when the text carries no letters at all. Whisper labels
Sanskrit as Hindi more often than not, so a Devanagari-dominant transcript is
treated as Sanskrit regardless of the hint. Any other script (Kannada, Urdu,
Arabic, ...) is "unknown", which the orchestrator refuses before the model
sees it.
"""

from __future__ import annotations

_DEVANAGARI = (0x0900, 0x097F)

SUPPORTED = ("en", "sa")


def detect(text: str, hint: str | None = None) -> str:
    counts = {"sa": 0, "en": 0}
    for ch in text:
        cp = ord(ch)
        if _DEVANAGARI[0] <= cp <= _DEVANAGARI[1]:
            counts["sa"] += 1
        elif ch.isascii() and ch.isalpha():
            counts["en"] += 1
    if sum(counts.values()) == 0:
        return hint if hint in SUPPORTED else "unknown"
    return max(counts, key=counts.get)  # type: ignore[arg-type]

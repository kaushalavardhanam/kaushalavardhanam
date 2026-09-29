"""Word/character error rates without an extra dependency."""

from __future__ import annotations

import re


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ɏऀ-ॿಕ-೯']+", (text or "").lower())


def levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    if isinstance(a, str):
        a = list(a)
    if isinstance(b, str):
        b = list(b)
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = _words(reference), _words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def cer(reference: str, hypothesis: str) -> float:
    ref = re.sub(r"\s+", "", (reference or "").lower())
    hyp = re.sub(r"\s+", "", (hypothesis or "").lower())
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def meaning_preserved(reference: str, hypothesis: str) -> bool:
    """Loose check: enough content words overlap, or WER is modest."""
    if wer(reference, hypothesis) <= 0.34:
        return True
    ref, hyp = set(_words(reference)), set(_words(hypothesis))
    if not ref:
        return not hyp
    return len(ref & hyp) / len(ref) >= 0.6

"""Deterministic output validation (FR-3.5, DESIGN §5).

Every reply passes here before it is spoken — the model is never trusted to
skip this guardrail (DESIGN §1.4). The retry-with-corrective-suffix policy
lives in the orchestrator so lexicon substitution can happen between
generation and validation.
"""

from __future__ import annotations

import re

_DEVANAGARI = (0x0900, 0x097F)

MAX_REPLY_CHARS = 220
MIN_DEVANAGARI_RATIO = 0.8

# Qwen's Devanagari training data is overwhelmingly Hindi, so it reaches for
# Hindi words under pressure — we logged "अहं आज किंचित् करिष्यामि" (आज for
# अद्य) and "अहं खेलानि करोमि" (खेल for क्रीडा). Both are 100% Devanagari, so
# the script ratio above waves them through; nothing else in the pipeline can
# see them either.
#
# Whole tokens only, and only words that are unambiguously NOT Sanskrit. Words
# the two languages share are deliberately absent: का is Hindi's genitive but
# also Sanskrit's feminine "which" (the corpus has भवतः वेतनश्रेणी का ?), या is
# Hindi "or" but Sanskrit's feminine relative pronoun, and कर is Hindi "do" but
# Sanskrit "hand". Flagging those would fail correct replies, and a false
# positive here costs a retry and possibly the safe fallback.
#
# Matching whole tokens also keeps Sanskrit infinitives safe: श्रोतुम् ends in
# तुम् but never tokenises to Hindi तुम.
_HINDI_MARKERS = frozenset("""
है हैं हूँ हूं हो था थी थे रहा रही रहे रहा
और नहीं क्या क्यों कैसे कहाँ कहां जब तब लेकिन
यह वह ये वे मैं मुझे हमें तुम तुम्हें तुम्हारा
में से को पर ने भी ही
कुछ बहुत अच्छा अच्छी ठीक सब
गया गई गए किया किये करता करती करते होता होती होते
सकता सकती सकते चाहिए वाला वाली
आज कल अभी यहाँ यहां वहाँ वहां
दोस्त बात काम
सुनोमि सुनोति सुनता सुनती सुनते सुनना
""".split())


# Hindi CONTENT words, matched by prefix so inflections are caught too:
# the model writes them with Sanskrit endings (मक्खनं, खेलानि), which whole-token
# matching above misses. This list is a stopgap and will stay whack-a-mole —
# every new Hindi noun needs a new entry. The real fix is the inverse check:
# look each word up in a Sanskrit morphological dictionary and reject what is
# not an attested form, which catches Hindi and invented words alike.
#
# Prefix matching is only safe where the stem cannot begin a Sanskrit word, so
# entries are added only after checking them against the verified corpus. Note
# what is deliberately ABSENT: सुन (Hindi "listen") would swallow सुन्दरम्, so
# its forms are listed whole in _HINDI_MARKERS instead.
_HINDI_STEMS = ("मक्खन", "खेल")


def devanagari_ratio(text: str) -> float:
    """Devanagari codepoints / all script codepoints (letters + combining marks).

    Whitespace, digits, and punctuation are ignored so danda and spaces don't
    dilute the ratio.
    """
    devanagari = 0
    other = 0
    for ch in text:
        if _DEVANAGARI[0] <= ord(ch) <= _DEVANAGARI[1]:
            devanagari += 1
        elif ch.isalpha():
            other += 1
    total = devanagari + other
    return devanagari / total if total else 0.0


def hindi_markers(text: str) -> list[str]:
    """Unambiguously-Hindi whole words in the reply (see _HINDI_MARKERS)."""
    # U+0964/0965 are the danda and double danda, and U+0966-096F are the
    # Devanagari digits — all inside the block, none of them word characters.
    # Including them glued the terminator onto the final token ("सुनोमि।"),
    # so the last word of every sentence silently escaped this check.
    tokens = re.findall(r"[\u0900-\u0963\u0972-\u097F]+", text)
    return [t for t in tokens
            if t in _HINDI_MARKERS or t.startswith(_HINDI_STEMS)]


def validate(text: str, max_chars: int = MAX_REPLY_CHARS,
             min_ratio: float = MIN_DEVANAGARI_RATIO) -> tuple[bool, str]:
    """Returns (ok, reason). Reason is "" when ok."""
    if not text or not text.strip():
        return False, "empty"
    if len(text) > max_chars:
        return False, f"too long ({len(text)} > {max_chars} chars)"
    ratio = devanagari_ratio(text)
    if ratio < min_ratio:
        return False, f"not Devanagari-dominant (ratio {ratio:.2f} < {min_ratio})"
    hindi = hindi_markers(text)
    if hindi:
        return False, f"Hindi, not Sanskrit ({', '.join(sorted(set(hindi)))})"
    return True, ""

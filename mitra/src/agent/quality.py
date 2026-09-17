"""Sanskrit *quality* checks above the Devanagari script gate (issue #9).

``validate()`` remains a script-only gate (Devanagari ratio / length). This
module flags Hindi contamination, unsegmentable tokens, person-agreement
errors, copular "I am dear X" templates, uninflected stems, and reuse of
the previous turn's content words.

``quality_ok`` is a separate signal from ``validation_ok``. Never treat
script validation as evidence of linguistic quality.
"""

from __future__ import annotations

import re
from typing import Iterable

_DEV_TOKEN = re.compile(r"[\u0900-\u097F]+")

# Whole-token Hindi / Hindi-derived forms that have a Sanskrit counterpart.
_HINDI_TOKENS = {
    "खेल", "खेलं", "खेलम्", "खेलति", "खेलनि",
    "खेलनीय", "खेलनीयं", "खेलनीया", "खेलनीये",
    "आज", "नहीं", "है", "हैं", "हूँ", "हूं",
    "क्या", "बहुत", "अच्छा", "कैसे", "कहाँ",
    "यहाँ", "वहाँ", "तुम्हां", "तुम्हारा", "तुम्हारे",
    "तुम्हें", "मिष्ठान",
}

# Substrings that make a token Hindi-contaminated even when mashed.
_HINDI_FRAGMENTS = ("तुम्हां", "खेल", "नहीं")

# Known unsegmentable / nonce tokens from the 2026-09-16 Mode A log.
_OOV_TOKENS = {
    "तुम्हांस्केत्रे",
    "मिष्ठान",
}

_UNINFLECTED_STEMS = {
    "मिष्ठान",  # should be मिष्टान्नम्
}

_THIRD_SG = (
    "करोति", "गच्छति", "वदति", "पठति", "करिष्यति",
    "क्रीडति", "शृणोति", "वसति", "तिष्ठति", "हसति",
)

_FIRST_SG = (
    "अस्मि", "करोमि", "वदामि", "पठामि", "क्रीडामि",
    "करिष्यामि", "वदिष्यामि", "शृणोमि", "वसामि",
    "गच्छामि", "जानामि", "अवगच्छामि",
)

_FUNCTION_WORDS = {
    "अहं", "अहम्", "अस्मि", "अस्ति", "भवान्", "भवतः", "भवता",
    "किम्", "कथम्", "न", "आम्", "मम", "त्वया", "तव", "सह",
    "मित्र", "मित्रम्", "एतत्", "इदानीं", "किञ्चित्",
}

_GREETING_CLOSE = re.compile(
    r"भवान्\s+कथम्|भवतः\s+किम्|भवतः\s+कथम्|भवान्\s+किम्"
)
_DEAR_X = re.compile(
    r"अह[ंम्]\s+.{0,40}?प्रिय[ंःाि]\s+अस्मि"
)
_AHAM = re.compile(r"अह[ंम्](?:\s|$)")

# Reject-class flags (quality_ok becomes False). Greeting is reported separately.
REJECT_FLAGS = (
    "hindi_lexeme",
    "oov_token",
    "person_agreement",
    "uninflected_stem",
    "copular_dear_x",
    "content_repeat",
)


def tokens(text: str) -> list[str]:
    return _DEV_TOKEN.findall(text or "")


def _content_tokens(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in _FUNCTION_WORDS and len(t) >= 3}


def evaluate_quality(
    text: str,
    *,
    previous_reply: str | None = None,
) -> dict:
    """Return quality flags for a Sanskrit reply.

    ``ok`` is False when any reject-class flag is present. Script validation
    is *not* performed here — call ``validator.validate`` separately.
    """
    flags: list[str] = []
    notes: list[str] = []
    found = tokens(text)

    hindi = sorted({
        t for t in found
        if t in _HINDI_TOKENS or any(frag in t for frag in _HINDI_FRAGMENTS)
    })
    if hindi:
        flags.append("hindi_lexeme")
        notes.append("Hindi/Hindi-derived lexeme: " + ", ".join(hindi))

    oov = sorted({t for t in found if t in _OOV_TOKENS or "तुम्हां" in t})
    mashed = [t for t in found if len(t) >= 10 and t not in _FUNCTION_WORDS
              and not any(t.endswith(suf) for suf in ("ामि", "ोमि", "अस्मि"))]
    # Long unsegmentable mash-ups (e.g. तुम्हांस्केत्रे) that are not verbs.
    oov = sorted(set(oov) | {t for t in mashed if any(f in t for f in _HINDI_FRAGMENTS)})
    if oov:
        if "oov_token" not in flags:
            flags.append("oov_token")
        notes.append("unsegmentable/OOV token: " + ", ".join(oov))

    stems = sorted({t for t in found if t in _UNINFLECTED_STEMS})
    if stems:
        flags.append("uninflected_stem")
        notes.append("uninflected stem in argument position: " + ", ".join(stems))

    if _AHAM.search(text or ""):
        has_3sg = any(v in (text or "") for v in _THIRD_SG)
        has_1sg = any(v in (text or "") for v in _FIRST_SG)
        if has_3sg and not has_1sg:
            flags.append("person_agreement")
            notes.append("first-person pronoun with third-person finite verb")

    if _DEAR_X.search(text or ""):
        flags.append("copular_dear_x")
        notes.append('copular "I am dear X" (अहं … प्रियं अस्मि)')

    if previous_reply:
        overlap = _content_tokens(text) & _content_tokens(previous_reply)
        if len(overlap) >= 2:
            flags.append("content_repeat")
            notes.append("reuses previous-turn content words: " + ", ".join(sorted(overlap)))

    greeting = bool(_GREETING_CLOSE.search(text or "") or "स्वागतम्" in (text or ""))
    if greeting:
        flags.append("greeting_template")
        notes.append("greeting template (स्वागतम् / भवान् कथम् / भवतः किम्)")

    reject = [f for f in flags if f in REJECT_FLAGS]
    return {
        "ok": not reject,
        "flags": flags,
        "reject_flags": reject,
        "notes": notes,
        "reason": "; ".join(notes),
    }


def evaluate_evidence_table(
    rows: Iterable[dict],
    *,
    target_ids: Iterable[int] | None = None,
) -> dict:
    """Precision/recall of reject-class flags against the issue #9 table.

    ``rows`` items: ``{id, sanskrit, previous, expect_reject}``.
    Default target is turns 1, 2, 3, 4, 5, 8, 9.
    """
    target = set(target_ids if target_ids is not None else (1, 2, 3, 4, 5, 8, 9))
    tp = fp = fn = tn = 0
    per: list[dict] = []
    for row in rows:
        prev = row.get("previous")
        result = evaluate_quality(row["sanskrit"], previous_reply=prev)
        predicted = not result["ok"]
        gold = int(row["id"]) in target
        if predicted and gold:
            tp += 1
        elif predicted and not gold:
            fp += 1
        elif (not predicted) and gold:
            fn += 1
        else:
            tn += 1
        per.append({
            "id": row["id"],
            "predicted_reject": predicted,
            "gold_reject": gold,
            "flags": result["flags"],
            "ok": result["ok"],
        })
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "target_ids": sorted(target),
        "rows": per,
    }

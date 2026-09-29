"""Sanskrit quality rubric (issue #7 acceptance).

Scale 1–5 for grammar and semantic correctness. Passing the Devanagari
validator is necessary but not sufficient.

Deterministic checks run here. Linguistic judgements are recorded with
justification and, when uncertain, flagged for human review. The cloud
agent (this session's model) is the primary evaluator and is not a
candidate model.
"""

from __future__ import annotations

import re

from mitra.agent.validator import devanagari_ratio, validate

# Common Hindi / Hinglish function words that should not appear in laukika Sanskrit replies.
_HINDI_MARKERS = (
    "है", "हैं", "हूँ", " हूं", "था", "थी", "थे", "क्या", "नहीं", "बहुत",
    "अच्छा", "कैसे", "कहाँ", "यहाँ", "वहाँ", "लिए", "बाद", "साथ में",
)

_LATIN = re.compile(r"[A-Za-z]{3,}")

PASS_THRESHOLD = 4  # exclusive floor for "needs corrected Sanskrit"
HARD_FLOOR = 3      # no controlled response may be below this


def script_language_score(text: str) -> tuple[int, str]:
    ok, reason = validate(text)
    ratio = devanagari_ratio(text)
    hindi = [m for m in _HINDI_MARKERS if m in (text or "")]
    latin = _LATIN.findall(text or "")
    if not ok:
        return 1, reason or "validator failed"
    if hindi:
        return 2, f"Hindi/mixed contamination: {hindi}"
    if latin:
        return 3, f"Latin words in a Sanskrit reply: {latin}"
    if ratio >= 0.98:
        return 5, f"Devanagari-dominant (ratio {ratio:.2f})"
    return 4, f"Devanagari with minor other script (ratio {ratio:.2f})"


def gloss_agrees(sanskrit: str, gloss: str | None) -> tuple[bool | None, str]:
    if not gloss or not gloss.strip():
        return None, "no English gloss"
    # Deterministic overlap is weak for Sanskrit↔English; flag for review
    # unless the gloss is empty or clearly Latin-only English.
    if devanagari_ratio(gloss) > 0.3:
        return False, "gloss is not English"
    return None, "gloss/Sanskrit agreement needs linguistic review"


def evaluate_response(
    *,
    prompt: str,
    sanskrit: str,
    gloss: str | None = None,
    grammar: int | None = None,
    semantic: int | None = None,
    naturalness: int | None = None,
    persona: int | None = None,
    justification: str = "",
    corrected: str | None = None,
    uncertain: bool = False,
    evaluator: str = "heuristic+human-or-cloud-agent",
    gloss_agrees_override: bool | None = None,
) -> dict:
    script, script_why = script_language_score(sanskrit)
    agree, agree_why = gloss_agrees(sanskrit, gloss)
    if gloss_agrees_override is not None:
        agree = gloss_agrees_override
        agree_why = "cloud-agent linguistic comparison of Sanskrit and English gloss"
    ok, val_reason = validate(sanskrit)
    result = {
        "prompt": prompt,
        "sanskrit": sanskrit,
        "gloss": gloss,
        "validator_ok": ok,
        "validator_reason": val_reason,
        "script_score": script,
        "script_note": script_why,
        "grammar": grammar,
        "semantic": semantic,
        "naturalness": naturalness,
        "persona": persona,
        "gloss_agrees": agree,
        "gloss_note": agree_why,
        "justification": justification,
        "corrected_sanskrit": corrected,
        "uncertain": uncertain,
        "evaluator": evaluator,
        "pass_threshold": PASS_THRESHOLD,
        "hard_floor": HARD_FLOOR,
    }
    result["needs_correction"] = (
        (grammar is not None and grammar < PASS_THRESHOLD)
        or (semantic is not None and semantic < PASS_THRESHOLD)
        or script < 4
    )
    result["hard_fail"] = (
        (grammar is not None and grammar < HARD_FLOOR)
        or (semantic is not None and semantic < HARD_FLOOR)
        or script < 3
        or not ok
    )
    return result


def aggregate(rows: list[dict]) -> dict:
    grammars = [r["grammar"] for r in rows if r.get("grammar") is not None]
    semantics = [r["semantic"] for r in rows if r.get("semantic") is not None]
    return {
        "n": len(rows),
        "grammar_mean": round(sum(grammars) / len(grammars), 2) if grammars else None,
        "semantic_mean": round(sum(semantics) / len(semantics), 2) if semantics else None,
        "all_devanagari": all(r.get("validator_ok") for r in rows),
        "any_hard_fail": any(r.get("hard_fail") for r in rows),
        "quality_gate": bool(
            rows
            and all(r.get("validator_ok") for r in rows)
            and grammars and semantics
            and (sum(grammars) / len(grammars)) >= 4.0
            and (sum(semantics) / len(semantics)) >= 4.0
            and all(g >= 3 for g in grammars)
            and all(s >= 3 for s in semantics)
        ),
    }

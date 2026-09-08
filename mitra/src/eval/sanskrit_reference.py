"""Cloud-agent reference Sanskrit for the ten conversation scenarios.

These are NOT Qwen or Bedrock outputs. They are the cloud agent's
(target-quality) replies used to (a) exercise the orchestrator/TTS path and
(b) show the rubric with justifications. Live candidate scores belong in
evals/results/ after a provider actually answers.
"""

from __future__ import annotations

# Evaluator identity for issue #7: must not be a candidate model judging itself.
EVALUATOR_ID = "cursor-grok-4.6-high-fast (cloud agent; not a Bedrock/Ollama candidate)"

REFERENCE_REPLIES: dict[str, dict] = {
    "doing": {
        "sanskrit": "अहं त्वया सह वदामि।",
        "gloss": "I am talking with you.",
        "grammar": 5,
        "semantic": 5,
        "naturalness": 5,
        "persona": 5,
        "justification": (
            "अहं (1sg nom) + त्वया (2sg inst) + सह + वदामि (1sg pres √वद्). "
            "Person and number agree; present tense matches the question; "
            "word order SOV; no Hindi markers."
        ),
    },
    "live": {
        "sanskrit": "अहं अत्र वसामि।",
        "gloss": "I live here.",
        "grammar": 5,
        "semantic": 4,
        "naturalness": 5,
        "persona": 5,
        "justification": (
            "अहं + अत्र + वसामि (1sg pres √वस्). Grammatically clean. Semantic "
            "4 rather than 5 only because a desktop robot could also say "
            "समीपे तिष्ठामि; meaning is still appropriate and honest."
        ),
    },
    "play": {
        "sanskrit": "आम्, अहं क्रीडामि।",
        "gloss": "Yes, I play.",
        "grammar": 5,
        "semantic": 5,
        "naturalness": 5,
        "persona": 4,
        "justification": (
            "आम् + अहं + क्रीडामि (1sg pres √क्रीड्). Agreement is correct. "
            "Persona 4: a robot 'playing' is slightly figurative but child-appropriate."
        ),
    },
    "food": {
        "sanskrit": "मम प्रियं भोजनं सेवफलम् अस्ति।",
        "gloss": "My favorite food is an apple.",
        "grammar": 4,
        "semantic": 4,
        "naturalness": 4,
        "persona": 4,
        "justification": (
            "मम (gen) प्रियं भोजनं सेवफलम् अस्ति — neuter nominatives agree. "
            "Minor style: a robot claiming a favorite food is playful, not factual. "
            "सेवफलम् is the verified lexicon name for apple."
        ),
        "uncertain": True,
    },
    "subject": {
        "sanskrit": "मम प्रियः विषयः संस्कृतम् अस्ति।",
        "gloss": "My favorite subject is Sanskrit.",
        "grammar": 4,
        "semantic": 5,
        "naturalness": 4,
        "persona": 5,
        "justification": (
            "विषयः m + प्रियः m agree. Predicate संस्कृतम् is a conventional "
            "neuter name of the language; slightly loose gender on the predicate "
            "but standard laukika. No Hindi. On-persona."
        ),
    },
    "reading": {
        "sanskrit": "अहं इदानीं न किञ्चित् पठामि।",
        "gloss": "I am not reading anything right now.",
        "grammar": 5,
        "semantic": 5,
        "naturalness": 5,
        "persona": 5,
        "justification": (
            "अहं + इदानीं + न + किञ्चित् + पठामि (1sg pres √पठ्). Honest, "
            "negation is Sanskrit न not Hindi नहीं. Word order natural."
        ),
    },
    "music": {
        "sanskrit": "आम्, अहं संगीतं शृणोमि।",
        "gloss": "Yes, I listen to music.",
        "grammar": 5,
        "semantic": 5,
        "naturalness": 5,
        "persona": 5,
        "justification": (
            "संगीतं n acc + शृणोमि (1sg pres √श्रु). Case for the object of "
            "hearing is accusative; person agrees."
        ),
    },
    "sports": {
        "sanskrit": "न, अहं क्रीडां न क्रीडामि। अहं मित्रम् अस्मि।",
        "gloss": "No, I do not play sports. I am Mitra.",
        "grammar": 4,
        "semantic": 4,
        "naturalness": 4,
        "persona": 5,
        "justification": (
            "Double न is clear if slightly heavy. क्रीडां (acc sg) + क्रीडामि. "
            "Second sentence अहं मित्रम् अस्मि is grammatically fine (neuter "
            "predicate noun). Avoids inventing a sports life."
        ),
    },
    "today": {
        "sanskrit": "अद्य अहं त्वया सह वदिष्यामि।",
        "gloss": "Today I will talk with you.",
        "grammar": 5,
        "semantic": 5,
        "naturalness": 5,
        "persona": 5,
        "justification": (
            "अद्य + अहं + त्वया सह + वदिष्यामि (1sg future √वद्). Tense matches "
            "'will you do today'; instrumentals with सह are correct."
        ),
    },
    "friend": {
        "sanskrit": "आम्, अहं तव मित्रम् अस्मि।",
        "gloss": "Yes, I am your friend.",
        "grammar": 5,
        "semantic": 5,
        "naturalness": 5,
        "persona": 5,
        "justification": (
            "आम् + अहं + तव (gen) + मित्रम् (n nom/acc used as predicate) + अस्मि. "
            "Answers the question directly; name/persona aligned."
        ),
    },
}


def as_eval_rows() -> list[dict]:
    from mitra.eval.sanskrit import evaluate_response

    rows = []
    for sid, data in REFERENCE_REPLIES.items():
        row = evaluate_response(
            prompt=sid,
            sanskrit=data["sanskrit"],
            gloss=data["gloss"],
            grammar=data["grammar"],
            semantic=data["semantic"],
            naturalness=data["naturalness"],
            persona=data["persona"],
            justification=data["justification"],
            uncertain=bool(data.get("uncertain")),
            evaluator=EVALUATOR_ID,
        )
        row["id"] = sid
        rows.append(row)
    return rows

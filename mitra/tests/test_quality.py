"""Quality stage is separate from the Devanagari script gate (issue #9)."""

from mitra.agent.quality import evaluate_evidence_table, evaluate_quality
from mitra.agent.validator import validate
from mitra.eval.qwen_mode_a_20260916 import TURNS
from mitra.eval.sanskrit_reference import REFERENCE_REPLIES


def _prev(n: int) -> str | None:
    if n <= 1:
        return None
    return TURNS[n - 2]["sanskrit"]


def test_script_gate_still_passes_the_failing_mode_a_turns():
    """The failure mode #7 warned about: Devanagari OK is not quality."""
    for turn in TURNS:
        ok, _ = validate(turn["sanskrit"])
        assert ok is True, turn["id"]


def test_quality_rejects_target_turns():
    rows = [
        {
            "id": t["n"],
            "sanskrit": t["sanskrit"],
            "previous": _prev(t["n"]),
        }
        for t in TURNS
    ]
    summary = evaluate_evidence_table(rows)
    assert summary["recall"] == 1.0
    assert summary["precision"] >= 0.7
    flagged = {r["id"] for r in summary["rows"] if r["predicted_reject"]}
    assert {1, 2, 3, 4, 5, 8, 9}.issubset(flagged)


def test_quality_flags_match_failure_classes():
    t1 = evaluate_quality(TURNS[0]["sanskrit"])
    assert "person_agreement" in t1["flags"]
    t2 = evaluate_quality(TURNS[1]["sanskrit"])
    assert "oov_token" in t2["flags"] or "hindi_lexeme" in t2["flags"]
    t3 = evaluate_quality(TURNS[2]["sanskrit"])
    assert "hindi_lexeme" in t3["flags"]
    t4 = evaluate_quality(TURNS[3]["sanskrit"])
    assert "copular_dear_x" in t4["flags"] or "uninflected_stem" in t4["flags"]
    t5 = evaluate_quality(TURNS[4]["sanskrit"])
    assert "copular_dear_x" in t5["flags"]
    t8 = evaluate_quality(TURNS[7]["sanskrit"])
    assert "hindi_lexeme" in t8["flags"]
    t9 = evaluate_quality(TURNS[8]["sanskrit"])
    assert "hindi_lexeme" in t9["flags"]


def test_reference_replies_pass_quality():
    prev = None
    for sid, row in REFERENCE_REPLIES.items():
        q = evaluate_quality(row["sanskrit"], previous_reply=prev)
        assert q["ok"], (sid, q)
        prev = row["sanskrit"]


def test_content_repeat_needs_previous_turn():
    q_alone = evaluate_quality(TURNS[5]["sanskrit"])
    q_with = evaluate_quality(TURNS[5]["sanskrit"], previous_reply=TURNS[4]["sanskrit"])
    assert "content_repeat" not in q_alone["flags"]
    assert "content_repeat" in q_with["flags"]


def test_greeting_template_is_flagged_but_not_required_to_reject():
    q = evaluate_quality("आम्, अहं तव मित्रम् अस्मि। भवतः कथम्?")
    assert "greeting_template" in q["flags"]
    assert q["ok"] is True


def test_quality_and_script_are_distinct_signals():
    text = TURNS[1]["sanskrit"]
    script_ok, _ = validate(text)
    q = evaluate_quality(text)
    assert script_ok is True
    assert q["ok"] is False

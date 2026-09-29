from mitra.agent.validator import validate
from mitra.eval.sanskrit import aggregate, evaluate_response, script_language_score
from mitra.eval.sanskrit_reference import EVALUATOR_ID, REFERENCE_REPLIES, as_eval_rows


def test_reference_replies_are_devanagari():
    for sid, row in REFERENCE_REPLIES.items():
        ok, reason = validate(row["sanskrit"])
        assert ok, (sid, reason, row["sanskrit"])


def test_hindi_contamination_is_flagged():
    score, note = script_language_score("मैं अच्छा हूँ।")
    assert score <= 2
    assert "Hindi" in note


def test_english_fallback_fails_script():
    score, _ = script_language_score("I am fine, thank you.")
    assert score == 1


def test_reference_set_meets_quality_gate():
    rows = as_eval_rows()
    summary = aggregate(rows)
    assert summary["all_devanagari"]
    assert summary["grammar_mean"] >= 4.0
    assert summary["semantic_mean"] >= 4.0
    assert not summary["any_hard_fail"]
    assert summary["quality_gate"]
    assert all(r["gloss_agrees"] is True for r in rows)
    assert "cloud agent" in EVALUATOR_ID.lower() or "grok" in EVALUATOR_ID.lower()


def test_evaluate_marks_needs_correction():
    row = evaluate_response(
        prompt="x", sanskrit="अहं अस्मि।", grammar=3, semantic=4,
        evaluator=EVALUATOR_ID,
    )
    assert row["needs_correction"] is True
    assert row["hard_fail"] is False


def test_evaluator_is_not_a_candidate_id():
    assert "nova" not in EVALUATOR_ID.lower()
    assert "qwen" not in EVALUATOR_ID.lower()
    assert "claude" not in EVALUATOR_ID.lower()

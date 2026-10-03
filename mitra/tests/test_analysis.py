from eval.analysis import (
    by_category,
    depth_trend,
    detect_issues,
    judge_gap,
    latin_fraction,
    rank_issues,
    recommendations,
    render_report,
    slope,
)


def _rec(text="नमस्ते मित्र", category="greeting", depth=0, grammar=0.9, hindi=False, **extra):
    rec = {
        "category": category,
        "depth": depth,
        "text": text,
        "grammar_score": grammar,
        "hindi_flag": hindi,
    }
    rec.update(extra)
    return rec


def test_latin_fraction():
    assert latin_fraction("नमस्ते") == 0.0
    assert latin_fraction("hello") == 1.0
    assert latin_fraction("") == 0.0


def test_detect_issues():
    assert detect_issues(_rec()) == []
    assert "empty" in detect_issues(_rec(text="  "))
    assert "latin" in detect_issues(_rec(text="नमस्ते hello world"))
    assert "hindi" in detect_issues(_rec(hindi=True))
    assert "low_grammar" in detect_issues(_rec(grammar=0.2))
    assert "repetition" in detect_issues(_rec(text="अ अ अ अ अ अ अ अ"))


def test_rank_issues_easiest_first():
    records = [_rec(text=""), _rec(grammar=0.1), _rec(text="hello नमस्ते")]
    order = [i["issue"] for i in rank_issues(records)]
    assert order.index("empty") < order.index("latin") < order.index("low_grammar")


def test_by_category_worst_first():
    records = [
        _rec(category="a", grammar=0.9),
        _rec(category="b", grammar=0.3),
    ]
    rows = by_category(records)
    assert rows[0]["category"] == "b"
    assert rows[0]["drag"] > 0


def test_slope_and_depth_trend():
    assert slope([0, 1, 2], [1.0, 0.5, 0.0]) == -0.5
    assert slope([0, 0], [1.0, 2.0]) is None
    records = [_rec(depth=0, grammar=0.9), _rec(depth=1, grammar=0.6), _rec(depth=2, grammar=0.3)]
    trend = depth_trend(records)
    assert trend["grammar_slope"] < 0
    assert any("depth" in r for r in recommendations(records))


def test_judge_gap():
    records = [_rec(judge_score=0.5, grammar=0.8), _rec(grammar=0.7)]
    rows = judge_gap(records)
    assert len(rows) == 1
    assert abs(rows[0]["gap"] - 0.3) < 1e-9


def test_empty_records():
    assert recommendations([]) == []
    assert render_report([]) == "No records to analyze.\n"
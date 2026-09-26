from mitra.agent.validator import devanagari_ratio, validate

PURE_SA = "एतत् सेवफलम् अस्ति।"
MIXED = "This is एतत् apple फलम् basically"
ENGLISH = "Sorry, I can only speak English."


def test_pure_devanagari_ratio_is_one():
    assert devanagari_ratio(PURE_SA) == 1.0


def test_english_ratio_is_zero():
    assert devanagari_ratio(ENGLISH) == 0.0


def test_punctuation_and_spaces_ignored():
    assert devanagari_ratio("नमस्ते! । ॥ 123") == 1.0


def test_valid_reply_passes():
    ok, reason = validate(PURE_SA)
    assert ok and reason == ""


def test_empty_fails():
    assert not validate("")[0]
    assert not validate("   ")[0]


def test_english_fails():
    ok, reason = validate(ENGLISH)
    assert not ok and "Devanagari" in reason


def test_mixed_below_threshold_fails():
    assert not validate(MIXED)[0]


def test_too_long_fails():
    ok, reason = validate("नमस्ते " * 40)
    assert not ok and "long" in reason


def test_custom_max_chars():
    assert validate(PURE_SA, max_chars=5)[0] is False


def test_hindi_words_fail_despite_pure_devanagari():
    """Devanagari script alone does not make it Sanskrit — the observed
    failure was अहं आज किंचित् करिष्यामि (आज is Hindi for अद्य)."""
    ok, reason = validate("अहं आज किंचित् करिष्यामि।")
    assert not ok and "Hindi" in reason


def test_hindi_marker_named_in_reason():
    assert "आज" in validate("अहं आज पठामि।")[1]


def test_sanskrit_words_shared_with_hindi_are_not_flagged():
    """का is Hindi's genitive but also Sanskrit's feminine "which"; या is
    Hindi "or" but Sanskrit's relative pronoun. Flagging them would fail
    correct replies."""
    assert validate("भवतः वेतनश्रेणी का?")[0]
    assert validate("एषा मम सखी या अस्ति।")[0]


def test_sanskrit_infinitive_is_not_hindi_tum():
    """श्रोतुम् ends in तुम् — whole-token matching must not catch it."""
    assert validate("अहं संगीतं श्रोतुम् इच्छामि।")[0]


def test_hindi_markers_helper_returns_words():
    from mitra.agent.validator import hindi_markers

    assert hindi_markers("अहं आज पठामि।") == ["आज"]
    assert hindi_markers("अहं पुस्तकं पठामि।") == []


def test_hindi_word_before_danda_is_caught():
    """Regression: the token regex once included U+0964, gluing the danda onto
    the final token so the last word of every sentence escaped the check."""
    ok, reason = validate("अहं गीतानि सुनोमि।")
    assert not ok and "सुनोमि" in reason


def test_hindi_stems_catch_sanskrit_inflections():
    """The model gives Hindi nouns Sanskrit endings — मक्खनं, खेलानि."""
    assert not validate("अहं मक्खनं प्रियम् अस्मि।")[0]
    assert not validate("अहं खेलानि करोमि।")[0]


def test_stem_matching_does_not_swallow_sanskrit_words():
    """सुन is deliberately absent from the stem list: it would catch सुन्दरम्."""
    assert validate("सुन्दरम् एतत् अस्ति।")[0]
    assert validate("अहं संगीतं शृणोमि।")[0]


def test_correct_favourite_constructions_pass():
    assert validate("मम प्रियं भोजनं नवनीतम् अस्ति।")[0]
    assert validate("मह्यं गणितं रोचते।")[0]

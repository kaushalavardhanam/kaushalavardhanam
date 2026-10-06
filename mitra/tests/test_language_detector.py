from mitra.language_detector import detect


def test_english():
    assert detect("What is your name?") == "en"


def test_other_scripts_are_unknown():
    # Kannada support was dropped; Whisper's Urdu/Arabic renderings of
    # Sanskrit must not reach the model either.
    assert detect("ನಿನ್ನ ಹೆಸರೇನು?") == "unknown"
    assert detect("اہل سمسکرتم ودا میں") == "unknown"


def test_sanskrit_devanagari():
    assert detect("किम् एतत्?") == "sa"


def test_majority_wins_in_mixed_text():
    assert detect("ok किम् एतत् वद माम्") == "sa"


def test_empty_uses_hint():
    assert detect("", hint="sa") == "sa"
    assert detect("", hint="kn") == "unknown"
    assert detect("", hint="fr") == "unknown"
    assert detect("123 !!") == "unknown"

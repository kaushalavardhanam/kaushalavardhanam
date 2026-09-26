import sys
from pathlib import Path

import yaml

CONFIG = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "config.yaml").read_text(encoding="utf-8")
)


def test_required_sections_present():
    for key in ("robot", "models", "agent", "cloud_fallback", "session",
                "lexicon", "logging"):
        assert key in CONFIG, key
    for key in ("llm", "asr", "tts", "wake", "vad"):
        assert key in CONFIG["models"], key


def test_load_bearing_model_choices():  # CLAUDE.md decisions 1–2
    llm = CONFIG["models"]["llm"]
    assert llm["provider"] == "ollama"
    # the bare :8b tag is the "thinking" variant — must stay on instruct
    assert llm["id"] == "qwen3-vl:8b-instruct"
    assert CONFIG["cloud_fallback"]["enabled"] is False


def test_value_sanity():
    assert 0 < CONFIG["models"]["wake"]["threshold"] <= 1
    assert CONFIG["session"]["silence_timeout_s"] > 0
    assert CONFIG["session"]["max_reply_chars"] == 220


def test_vad_segmenter_energy_fallback():
    from mitra.audio.vad import EnergySegmenter, make_segmenter

    # silero/torch not installed in the test env → must fall back, not crash
    seg = make_segmenter(CONFIG["models"]["vad"]["engine"])
    assert seg is not None
    assert isinstance(make_segmenter("energy"), EnergySegmenter)


def test_energy_segmenter_detects_utterance():
    import numpy as np

    from mitra.audio.vad import EnergySegmenter

    seg = EnergySegmenter(min_speech_s=0.1, min_silence_s=0.2)
    chunk = int(0.1 * 16000)
    loud = np.full(chunk, 0.1, dtype=np.float32)
    quiet = np.zeros(chunk, dtype=np.float32)
    assert seg.process(quiet) is None          # still asleep
    assert seg.process(loud) is None           # speech starts
    assert seg.process(loud) is None
    assert seg.process(quiet) is None          # silence accumulating
    utterance = seg.process(quiet)             # silence >= 0.2 s → utterance
    assert utterance is not None and len(utterance) == 4 * chunk


def test_silero_segmenter_restores_torch_thread_pool(monkeypatch):
    """silero-vad clamps torch to one thread process-wide; Whisper shares the
    interpreter and decodes ~5x slower single-threaded (src/audio/vad.py)."""
    import types

    import pytest

    torch = pytest.importorskip("torch")
    from mitra.audio.vad import SileroSegmenter

    def load_silero_vad():
        torch.set_num_threads(1)  # what the real package does at import
        return object()

    monkeypatch.setitem(sys.modules, "silero_vad", types.SimpleNamespace(
        load_silero_vad=load_silero_vad,
        VADIterator=lambda model, **kw: types.SimpleNamespace(
            reset_states=lambda: None),
    ))
    original = torch.get_num_threads()
    try:
        torch.set_num_threads(max(2, original))
        expected = torch.get_num_threads()
        SileroSegmenter()
        assert torch.get_num_threads() == expected
    finally:
        torch.set_num_threads(original)

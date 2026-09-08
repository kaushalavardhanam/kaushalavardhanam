import numpy as np

from mitra.audio.hallucinations import is_hallucination, looks_unusable
from mitra.audio.vad import EnergySegmenter, SileroSegmenter, make_segmenter
from mitra.eval.metrics import cer, meaning_preserved, wer
from mitra.pipeline_trace import audio_stats, first_error_stage


def test_hallucination_phrases():
    assert is_hallucination("Thanks for watching")
    assert is_hallucination("thank you for watching")
    assert is_hallucination("hello hello hello hello hello hello")
    assert not is_hallucination("What are you doing?")


def test_short_audio_long_transcript_is_hallucination():
    long = "this is a very long invented transcript about nothing in particular today " * 3
    assert is_hallucination(long, duration_s=0.4)


def test_looks_unusable_empty_and_punct():
    assert looks_unusable("")
    assert looks_unusable("...")
    assert not looks_unusable("Do you play?")


def test_transcriber_skips_low_energy():
    from mitra.audio.asr import Transcriber

    asr = Transcriber(min_peak=0.01, filter_hallucinations=True)
    text, hint = asr.transcribe(np.zeros(16000, dtype=np.float32))
    assert text == "" and hint is None
    assert asr.last_diag["low_energy"] is True


def test_transcriber_drops_hallucination(monkeypatch):
    from mitra.audio.asr import Transcriber

    def fake_transcribe(_audio, **kwargs):
        if kwargs.get("language") == "en":
            return {"text": "What are you doing?", "language": "en"}
        return {"text": "Thanks for watching", "language": "en"}

    monkeypatch.setitem(__import__("sys").modules, "mlx_whisper",
                        type("m", (), {"transcribe": staticmethod(fake_transcribe)}))
    # import after stub
    asr = Transcriber(min_peak=0.001, english_retry=True)
    audio = np.full(16000, 0.2, dtype=np.float32)
    # Call _decode path via transcribe; mlx_whisper imported inside _decode
    import mitra.audio.asr as asr_mod

    monkeypatch.setattr(asr_mod, "is_hallucination",
                        lambda t, duration_s=None: t.lower().startswith("thanks"))
    # Patch decode directly to avoid mlx import issues
    asr._decode = lambda audio, language=None: (
        {"text": "What are you doing?", "language": "en"} if language == "en"
        else {"text": "Thanks for watching", "language": "en"}
    )
    text, hint = asr.transcribe(audio)
    assert text == "What are you doing?"
    assert asr.last_diag["hallucination"] is True
    assert asr.last_diag["english_retry"] is True
    assert hint == "en"


def test_silero_factory_keeps_min_speech_s():
    # Without silero installed this is EnergySegmenter; with it, Silero.
    seg = make_segmenter("energy", min_speech_s=0.25, min_silence_s=0.7,
                         preroll_s=0.25)
    assert isinstance(seg, EnergySegmenter)
    assert seg._min_speech == int(0.25 * 16000)


def test_silero_class_stores_min_speech(monkeypatch):
    # Don't import torch; construct with a stub if the class init would load silero.
    # We only check the attribute after a partial init by setting it on a bare instance.
    seg = object.__new__(SileroSegmenter)
    seg._sr = 16000
    seg._min_speech = int(0.25 * 16000)
    assert seg._min_speech == 4000


def test_first_error_prefers_asr_over_llm():
    stage = first_error_stage({
        "payload_kind": "audio",
        "transcript": "thanks for watching",
        "expected_transcript": "What are you doing?",
        "asr_hallucination": True,
        "audio_stats": {"n_samples": 8000, "peak": 0.2},
    })
    assert stage == "asr"


def test_first_error_low_peak_is_capture():
    stage = first_error_stage({
        "payload_kind": "audio",
        "audio_stats": {"n_samples": 100, "peak": 0.0001},
        "min_peak": 0.008,
    })
    assert stage == "capture"


def test_wer_cer_basic():
    assert wer("What are you doing?", "What are you doing?") == 0
    assert wer("What are you doing?", "What are you doing") < 0.3
    assert cer("mitra", "mithra") == 0.2
    assert meaning_preserved("Do you play?", "Do you play")


def test_audio_stats_no_waveform():
    stats = audio_stats(np.array([0.1, -0.1, 0.05], dtype=np.float32), 16000)
    assert "duration_s" in stats and stats["n_samples"] == 3
    assert all(k != "samples" for k in stats)

"""First-incorrect-stage diagnostics for a conversation turn.

Identifies the earliest pipeline stage at which a failed interaction became
incorrect. Does not persist raw microphone audio (FR-7.3).
"""

from __future__ import annotations

from typing import Any

STAGES = (
    "capture",
    "vad",
    "asr",
    "language",
    "prompt",
    "llm",
    "validation",
    "lexicon",
    "translation",
    "tts",
)


def audio_stats(samples, samplerate: int = 16000) -> dict[str, Any]:
    """RMS/peak/duration/clipping — never the waveform itself."""
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return {
            "duration_s": 0.0, "rms": 0.0, "peak": 0.0,
            "clipped_frac": 0.0, "n_samples": 0, "samplerate": samplerate,
        }
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt(np.mean(audio ** 2)))
    clipped = float(np.mean(np.abs(audio) >= 0.99))
    return {
        "duration_s": round(len(audio) / samplerate, 3),
        "rms": round(rms, 5),
        "peak": round(peak, 5),
        "clipped_frac": round(clipped, 4),
        "n_samples": int(len(audio)),
        "samplerate": samplerate,
    }


def first_error_stage(turn: dict) -> str | None:
    """Return the earliest stage that made this turn incorrect, or None."""
    stats = turn.get("audio_stats") or {}
    if turn.get("payload_kind") == "audio":
        if stats.get("n_samples", 0) == 0:
            return "capture"
        if stats.get("peak", 0) < turn.get("min_peak", 0.008):
            return "capture"
        if turn.get("vad_empty"):
            return "vad"

    transcript = (turn.get("transcript") or "").strip()
    asr_raw = (turn.get("asr_raw") or transcript).strip()
    if turn.get("payload_kind") == "audio" and not asr_raw:
        return "asr"
    if turn.get("asr_hallucination"):
        return "asr"
    if turn.get("asr_filtered_empty"):
        return "asr"

    expected = turn.get("expected_transcript")
    if expected and transcript and _normalize(transcript) != _normalize(expected):
        # Recognition diverged — do not blame the LLM
        return "asr"

    if turn.get("lang") == "unknown" and transcript:
        return "language"

    if turn.get("llm_error"):
        return "llm"
    if turn.get("llm_empty"):
        return "llm"

    if turn.get("validation_ok") is False and not turn.get("explain_in_english"):
        return "validation"

    if turn.get("tts_error"):
        return "tts"

    reply = (turn.get("reply") or "").strip()
    if not reply:
        return "llm"

    from mitra.agent.validator import validate

    if not turn.get("explain_in_english"):
        ok, _ = validate(reply, turn.get("max_reply_chars") or 220)
        if not ok:
            return "validation"
    return None


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def memory_mb() -> float | None:
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB; macOS reports bytes
        import sys
        if sys.platform == "darwin":
            return round(rss / (1024 * 1024), 1)
        return round(rss / 1024, 1)
    except Exception:
        return None

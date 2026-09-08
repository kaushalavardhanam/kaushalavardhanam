"""Local ASR (FR-4.2): Whisper via mlx-whisper for en/kn, with an optional
Sanskrit fine-tune rescue pass.

Whisper has no Sanskrit language code — Devanagari output usually comes back
tagged "hi". When the transcript is Devanagari-dominant and a Sanskrit model is
configured, the audio is re-transcribed with the fine-tune and tagged "sa"
(experimental, REQUIREMENTS R7).

Recognition diagnostics (issue #7): peak-normalize only when the capture is
loud enough to be speech; drop known Whisper hallucinations; retry a
garbled first pass with an English language hint. These are ASR-stage
fixes — they do not change the LLM.
"""

from __future__ import annotations

import logging

import numpy as np

from mitra import language_detector
from mitra.audio.hallucinations import is_hallucination, looks_unusable
from mitra.pipeline_trace import audio_stats

logger = logging.getLogger("mitra")

# Quiet leftover after a flush, or laptop-fan noise, must not be amplified
# into a Whisper hallucination by peak-normalizing toward 0.9.
DEFAULT_MIN_PEAK = 0.008
DEFAULT_INITIAL_PROMPT = (
    "Mitra. Short conversational questions in English, Kannada, or Sanskrit."
)


class Transcriber:
    def __init__(self, default_model: str = "mlx-community/whisper-large-v3-mlx",
                 sanskrit_model: str | None = None, backend: str = "mlx",
                 device: str = "mps",
                 initial_prompt: str | None = DEFAULT_INITIAL_PROMPT,
                 condition_on_previous_text: bool = False,
                 no_speech_threshold: float = 0.6,
                 compression_ratio_threshold: float = 2.4,
                 logprob_threshold: float = -1.0,
                 min_peak: float = DEFAULT_MIN_PEAK,
                 filter_hallucinations: bool = True,
                 english_retry: bool = True):
        if backend != "mlx":
            raise ValueError(f"unsupported ASR backend: {backend!r} (v1 uses mlx)")
        self._default_model = default_model
        self._sanskrit_model = sanskrit_model
        self._device = device
        self._sa_pipeline = None  # lazy: only load if Sanskrit is actually spoken
        self._initial_prompt = initial_prompt
        self._condition_on_previous_text = condition_on_previous_text
        self._no_speech_threshold = no_speech_threshold
        self._compression_ratio_threshold = compression_ratio_threshold
        self._logprob_threshold = logprob_threshold
        self.min_peak = min_peak
        self._filter_hallucinations = filter_hallucinations
        self._english_retry = english_retry
        self.last_diag: dict = {}

    def transcribe(self, audio_16k_mono: np.ndarray) -> tuple[str, str | None]:
        """Returns (transcript, language hint from the ASR engine)."""
        audio = np.asarray(audio_16k_mono, dtype=np.float32).reshape(-1)
        stats = audio_stats(audio, 16000)
        self.last_diag = {"audio_stats": stats, "hallucination": False,
                          "english_retry": False, "low_energy": False}

        peak = stats["peak"]
        if peak < self.min_peak:
            self.last_diag["low_energy"] = True
            logger.info("asr: low-energy capture (peak=%.5f < %.5f) — skipping decode",
                        peak, self.min_peak)
            return "", None

        # Normalize only when there is real headroom; tiny peaks are noise.
        work = audio / peak * 0.9

        result = self._decode(work, language=None)
        text = (result.get("text") or "").strip()
        lang = result.get("language")
        duration_s = stats["duration_s"]

        if self._filter_hallucinations and is_hallucination(text, duration_s=duration_s):
            self.last_diag["hallucination"] = True
            logger.info("asr: dropped hallucination %r (%.2fs audio)", text, duration_s)
            text = ""

        if self._english_retry and looks_unusable(text):
            retry = self._decode(work, language="en")
            retry_text = (retry.get("text") or "").strip()
            if retry_text and not is_hallucination(retry_text, duration_s=duration_s):
                self.last_diag["english_retry"] = True
                logger.info("asr: english retry recovered %r (was %r)", retry_text, text)
                text, lang = retry_text, retry.get("language") or "en"

        if self._sanskrit_model and text and language_detector.detect(text, lang) == "sa":
            try:
                text, lang = self._transcribe_sanskrit(audio), "sa"
            except Exception:  # experimental path must not break the turn (R7)
                pass

        self.last_diag["transcript"] = text
        self.last_diag["asr_hint"] = lang
        return text, lang

    def _decode(self, audio: np.ndarray, language: str | None) -> dict:
        import mlx_whisper

        kwargs = {
            "path_or_hf_repo": self._default_model,
            "condition_on_previous_text": self._condition_on_previous_text,
            "no_speech_threshold": self._no_speech_threshold,
            "compression_ratio_threshold": self._compression_ratio_threshold,
            "logprob_threshold": self._logprob_threshold,
        }
        if language:
            kwargs["language"] = language
        if self._initial_prompt:
            kwargs["initial_prompt"] = self._initial_prompt
        try:
            return mlx_whisper.transcribe(audio, **kwargs)
        except TypeError:
            # Older mlx-whisper builds reject some decoding kwargs
            fallback = {"path_or_hf_repo": self._default_model}
            if language:
                fallback["language"] = language
            return mlx_whisper.transcribe(audio, **fallback)

    def _transcribe_sanskrit(self, audio: np.ndarray) -> str:
        if self._sa_pipeline is None:
            from transformers import pipeline

            self._sa_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self._sanskrit_model,
                device=self._device,
            )
        out = self._sa_pipeline(
            {"array": np.asarray(audio, dtype=np.float32), "sampling_rate": 16000}
        )
        return out["text"].strip()

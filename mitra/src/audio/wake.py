"""Wake-word detection for "mitra" (FR-1) — मित्र, the vocative (सम्बोधन
विभक्ति) of मित्रम्, is how you *call* the robot.

Two engines behind one ``process(chunk) -> bool`` interface:

- ``TranscriptWakeDetector`` (default): an energy gate buffers short speech
  windows and Whisper transcribes them; wake fires when the phrase appears.
  No training needed. Uses whisper-**small** rather than tiny/base — testing
  against real recordings showed tiny mis-hears "mitra" and base can
  hallucinate repeated words on short clips (a known Whisper failure mode);
  small was the smallest size that reliably caught it. Costs a bit more
  latency/CPU than a trained wake-word model, in exchange for zero training.
  Two ASR backends, matching src/audio/asr.py: ``mlx`` (Apple Silicon) and
  ``transformers`` (everywhere else, incl. Linux).
- ``WakeWordDetector``: openWakeWord, the production target once the custom
  "mitra" onnx model is trained (Phase 1: synthetic speaker/accent/noise
  variants, FR-1.4 accuracy targets).
"""

from __future__ import annotations

import logging
import re

import numpy as np

from .vad import EnergySegmenter

logger = logging.getLogger("mitra")


class WakeWordDetector:
    def __init__(self, model: str = "hey_jarvis_v0.1", threshold: float = 0.6):
        try:
            from openwakeword.model import Model
        except ImportError as e:
            raise ImportError(
                "openwakeword is required for wake-word detection. "
                "Install with: pip install 'mitra[wake]' (or openwakeword)"
            ) from e
        # tflite-runtime has no Apple Silicon wheels — use the onnx backend.
        try:
            self._model = Model(wakeword_models=[model], inference_framework="onnx")
        except Exception:
            # Pretrained model names need a one-time asset download. onnxruntime
            # raises its own exception type for a missing file, so catch broadly
            # and let the retry surface any real error.
            from openwakeword.utils import download_models

            download_models()
            self._model = Model(wakeword_models=[model], inference_framework="onnx")
        self._threshold = threshold

    def process(self, chunk_16k_mono: np.ndarray) -> bool:
        """Score one 16 kHz mono float32 chunk; True on detection."""
        if len(chunk_16k_mono) == 0:
            return False
        pcm = (np.clip(chunk_16k_mono, -1.0, 1.0) * 32767).astype(np.int16)
        scores = self._model.predict(pcm)
        return max(scores.values()) >= self._threshold

    def reset(self) -> None:
        self._model.reset()


class TranscriptWakeDetector:
    """ASR-transcript wake matching — works without a trained wake model.

    Whisper spells the word many ways ("Mitra", "Mithra", "मित्र" …), so the
    match is against normalized variants with punctuation/spacing stripped.
    """

    def __init__(self, phrase: str = "mitra",
                 asr_model: str = "mlx-community/whisper-small-mlx",
                 backend: str = "mlx",
                 device: str = "cpu",
                 energy_threshold: float | None = None,  # None = adaptive gate
                 min_speech_s: float = 0.3,
                 hangover_s: float = 0.5, max_window_s: float = 3.0,
                 transcribe_fn=None):
        if backend not in ("mlx", "transformers"):
            raise ValueError(f"unsupported wake ASR backend: {backend!r}")
        base = phrase.strip().lower()
        self._variants = {base, base.replace("t", "th"), "मित्र"}
        self._asr_model = asr_model
        self._device = device
        self._pipeline = None  # lazy: transformers backend
        default_fn = self._mlx_transcribe if backend == "mlx" else self._hf_transcribe
        self._transcribe = transcribe_fn or default_fn
        self._segmenter = EnergySegmenter(
            threshold=energy_threshold, min_speech_s=min_speech_s,
            min_silence_s=hangover_s, max_utterance_s=max_window_s,
        )

    def _mlx_transcribe(self, audio: np.ndarray) -> str:
        import mlx_whisper

        peak = float(np.abs(audio).max())
        if peak > 0:  # normalize: Whisper mis-hears quiet capture badly
            audio = (audio / peak * 0.9).astype(np.float32)
        return mlx_whisper.transcribe(
            audio, path_or_hf_repo=self._asr_model
        ).get("text", "")

    def _hf_transcribe(self, audio: np.ndarray) -> str:
        # Lazy for the same reason as the mlx path: warmup() is called after
        # main.py logs "warming up wake ASR", so the download happens visibly.
        # Keep this on CPU — it runs on every gated speech window, forever, and
        # would otherwise sit resident in VRAM competing with the LLM.
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model=self._asr_model,
                device=self._device,
            )
        peak = float(np.abs(audio).max())
        if peak > 0:  # normalize: Whisper mis-hears quiet capture badly
            audio = (audio / peak * 0.9).astype(np.float32)
        out = self._pipeline(
            {"array": np.asarray(audio, dtype=np.float32), "sampling_rate": 16000},
            generate_kwargs={"language": "en", "task": "transcribe"},
        )
        return out.get("text", "")

    def process(self, chunk_16k_mono: np.ndarray) -> bool:
        utterance = self._segmenter.process(chunk_16k_mono)
        if utterance is None:
            return False
        text = self._transcribe(utterance)
        cleaned = re.sub(r"[^\wऀ-ॿ]+", "", text.lower())
        woke = any(v in cleaned for v in self._variants)
        # Visible feedback in --debug: without this, a failed wake check is
        # indistinguishable from "heard nothing at all".
        logger.debug("wake check (%.1fs audio) heard %r -> %s",
                     len(utterance) / 16000, text.strip(),
                     "WAKE" if woke else "no match")
        return woke

    def warmup(self) -> None:
        """Trigger the one-time whisper download/load before the run loop, so
        the first real wake attempt isn't stalled behind it."""
        self._transcribe(np.zeros(8000, dtype=np.float32))

    def reset(self) -> None:
        self._segmenter.reset()


def make_wake_detector(engine: str = "asr", **cfg):
    if engine == "asr":
        return TranscriptWakeDetector(
            phrase=cfg.get("phrase", "mitra"),
            asr_model=cfg.get("asr_model", "mlx-community/whisper-tiny"),
            backend=cfg.get("backend", "mlx"),
            device=cfg.get("device", "cpu"),
        )
    if engine == "openwakeword":
        return WakeWordDetector(model=cfg.get("model", "models/mitra.onnx"),
                                threshold=cfg.get("threshold", 0.6))
    raise ValueError(f"unknown wake engine: {engine!r}")
"""Local ASR (FR-4.2): Whisper via mlx-whisper (mac) or transformers (linux),
with an optional Sanskrit fine-tune rescue pass.

Two backends behind one ``transcribe()``: ``mlx`` (Apple Silicon, fastest on
an M-series GPU) and ``transformers`` (everywhere else — mlx-whisper has no
non-Darwin wheels). Selected by ``models.asr.backend`` in config.yaml.

Whisper has no Sanskrit language code — Devanagari output usually comes back
tagged "hi". When the transcript is Devanagari-dominant and a Sanskrit model is
configured, the audio is re-transcribed with the fine-tune and tagged "sa"
(experimental, REQUIREMENTS R7).
"""

from __future__ import annotations

import numpy as np

from mitra import language_detector


class Transcriber:
    def __init__(self, default_model: str = "mlx-community/whisper-large-v3-mlx",
                 sanskrit_model: str | None = None, backend: str = "mlx",
                 device: str = "mps"):
        if backend not in ("mlx", "transformers"):
            raise ValueError(f"unsupported ASR backend: {backend!r}")
        self._backend = backend
        self._default_model = default_model
        self._sanskrit_model = sanskrit_model
        self._device = device
        self._sa_pipeline = None  # lazy: only load if Sanskrit is actually spoken
        self._hf_pipeline = None  # lazy: transformers backend, default model

    def transcribe(self, audio_16k_mono: np.ndarray) -> tuple[str, str | None]:
        """Returns (transcript, language hint from the ASR engine)."""
        audio = np.asarray(audio_16k_mono, dtype=np.float32)
        peak = float(np.abs(audio).max())
        if peak > 0:  # normalize: Whisper mis-hears quiet capture badly
            audio = audio / peak * 0.9

        if self._backend == "mlx":
            import mlx_whisper

            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self._default_model,
            )
            text = result.get("text", "").strip()
            lang = result.get("language")
        else:
            text, lang = self._transcribe_hf(audio)

        if self._sanskrit_model and language_detector.detect(text, lang) == "sa":
            try:
                text, lang = self._transcribe_sanskrit(audio_16k_mono), "sa"
            except Exception:  # experimental path must not break the turn (R7)
                pass
        return text, lang

    def _transcribe_hf(self, audio: np.ndarray) -> tuple[str, str | None]:
        # Built on first call, not in __init__: main.py logs a "warming up ASR"
        # line before the first transcribe, so the multi-hundred-MB load
        # happens where the user can see why the startup pauses.
        if self._hf_pipeline is None:
            from transformers import pipeline

            self._hf_pipeline = pipeline(
                "automatic-speech-recognition",
                model=self._default_model,
                device=self._device,
                chunk_length_s=30,
            )
        out = self._hf_pipeline(
            {"array": np.asarray(audio, dtype=np.float32), "sampling_rate": 16000},
            generate_kwargs={"language": "en", "task": "transcribe"},
            return_language=True,
        )
        text = out["text"].strip()
        # return_language surfaces the detected language per chunk, as a name
        # ("english") rather than the ISO code mlx returns. language_detector
        # falls back to script heuristics when the hint is unfamiliar, which is
        # the reliable path for our en/kn/sa mix anyway.
        chunks = out.get("chunks") or []
        lang = chunks[0].get("language") if chunks else None
        return text, lang

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
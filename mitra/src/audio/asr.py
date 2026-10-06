"""Local ASR (FR-4.2): Whisper via mlx-whisper (mac) or transformers (linux),
with an optional Sanskrit fine-tune rescue pass.

Two backends behind one ``transcribe()``: ``mlx`` (Apple Silicon, fastest on
an M-series GPU) and ``transformers`` (everywhere else — mlx-whisper has no
non-Darwin wheels). Selected by ``models.asr.backend`` in config.yaml.

Sanskrit-aware routing (``routing: sanskrit_aware``, the default)
-----------------------------------------------------------------
Whisper does have a Sanskrit token (``sa``), but its free language-ID almost
never picks it: on spoken Sanskrit it guesses en / ur / ar / vi and then
writes Latin, Urdu or Arabic script ("कः त्वम् असि" → "Ka Tamasti."). That
garbage reached the LLM tagged ``[lang=en]``. Forcing English made it worse
— it invents fluent English ("भवतः नाम किम्" → "What is the name of God?").

Mitra hears only English or Sanskrit, so language-ID is reduced to two
outcomes: P(en), and everything else pooled as "sa". Then:

* clearly English → decode ``en``;
* clearly not English → decode ``sa`` (Devanagari output);
* in between → decode both ``en`` and ``sa`` and keep the one Whisper is
  more confident in (mean ``avg_logprob``), preferring ``sa`` within a margin.

Each decode gets its own short prompt in its own script — the old shared
English prompt ("...English, Kannada, or Sanskrit.") was itself emitted as a
transcript on quiet audio, and is now also filtered as a prompt leak.
``scripts/asr_lid_probe.py`` is the measurement behind the thresholds.

When a Sanskrit fine-tune is configured (``models.asr.sanskrit``), it replaces
the Whisper ``sa`` decode (experimental, REQUIREMENTS R7).

``routing: auto`` keeps the previous behaviour (free language-ID plus an
English retry) for comparison.
"""

from __future__ import annotations

import difflib
import logging
import math
import re

import numpy as np

from mitra import language_detector
from mitra.audio.hallucinations import is_hallucination, looks_unusable
from mitra.pipeline_trace import audio_stats

logger = logging.getLogger("mitra")

# Quiet leftover after a flush, or laptop-fan noise, must not be amplified
# into a Whisper hallucination by peak-normalizing toward 0.9.
DEFAULT_MIN_PEAK = 0.008
# Per-language decoding prompts. Short on purpose: Whisper copies long prompts
# into the transcript on near-silence. The Sanskrit one is Devanagari so the
# decoder is primed for the script and for word endings (visarga, anusvara,
# virama) it otherwise drops.
DEFAULT_INITIAL_PROMPT = "Hello Mitra."
DEFAULT_SANSKRIT_PROMPT = "नमस्ते मित्र। भवतः नाम किम्? अहं कुशली अस्मि।"

ROUTINGS = ("sanskrit_aware", "auto")

# Routing thresholds over restricted LID (see module docstring).
DEFAULT_EN_CONFIDENCE = 0.80   # P(en) at or above → English only
DEFAULT_SA_CONFIDENCE = 0.60   # P(indic) at or above → Sanskrit only
DEFAULT_SA_MARGIN = 0.25       # ambiguous: keep sa unless en beats it by this


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower(), flags=re.UNICODE).strip()


def is_prompt_leak(text: str, prompt: str | None) -> bool:
    """True when the 'transcript' is just the decoding prompt echoed back."""
    t, p = _norm(text or ""), _norm(prompt or "")
    if not t or not p:
        return False
    # A leak reproduces most of the prompt. A short phrase that merely also
    # appears in it ("भवतः नाम किम्") is a real utterance and must survive.
    if t in p and len(t) >= 0.6 * len(p):
        return True
    return difflib.SequenceMatcher(None, t, p).ratio() >= 0.8


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
                 english_retry: bool = True,
                 cpu_model: str | None = None,
                 routing: str = "sanskrit_aware",
                 sanskrit_prompt: str | None = DEFAULT_SANSKRIT_PROMPT,
                 en_confidence: float = DEFAULT_EN_CONFIDENCE,
                 sa_confidence: float = DEFAULT_SA_CONFIDENCE,
                 sa_margin: float = DEFAULT_SA_MARGIN):
        if backend not in ("mlx", "auto", "openai", "openai-whisper"):
            raise ValueError(
                f"unsupported ASR backend: {backend!r} "
                "(mlx | auto | openai-whisper)"
            )
        if routing not in ROUTINGS:
            raise ValueError(f"unsupported ASR routing: {routing!r} ({' | '.join(ROUTINGS)})")
        self._routing = routing
        self._sanskrit_prompt = sanskrit_prompt
        self._en_confidence = en_confidence
        self._sa_confidence = sa_confidence
        self._sa_margin = sa_margin
        self._backend = backend
        self._default_model = default_model
        self._sanskrit_model = sanskrit_model
        self._device = device
        self._cpu_model = cpu_model
        self._resolved_backend: str | None = None
        self._openai_model = None
        self._sa_pipeline = None  # lazy: only load if Sanskrit is actually spoken
        self._hf_pipeline = None  # lazy: transformers backend, default model
        self._initial_prompt = initial_prompt
        self._condition_on_previous_text = condition_on_previous_text
        self._no_speech_threshold = no_speech_threshold
        self._compression_ratio_threshold = compression_ratio_threshold
        self._logprob_threshold = logprob_threshold
        self.min_peak = min_peak
        self._filter_hallucinations = filter_hallucinations
        self._english_retry = english_retry
        self.last_diag: dict = {}

    @classmethod
    def from_config(cls, asr_cfg: dict, **overrides) -> "Transcriber":
        """One place that maps ``models.asr`` keys to constructor args
        (main.py, eval_recognition.py and test_audio.py all build from this)."""
        kw = dict(
            default_model=asr_cfg.get("default", "mlx-community/whisper-large-v3-turbo"),
            sanskrit_model=asr_cfg.get("sanskrit"),
            backend=asr_cfg.get("backend", "mlx"),
            device=asr_cfg.get("device", "mps"),
            initial_prompt=asr_cfg.get("initial_prompt", DEFAULT_INITIAL_PROMPT),
            sanskrit_prompt=asr_cfg.get("sanskrit_prompt", DEFAULT_SANSKRIT_PROMPT),
            condition_on_previous_text=asr_cfg.get("condition_on_previous_text", False),
            no_speech_threshold=asr_cfg.get("no_speech_threshold", 0.6),
            compression_ratio_threshold=asr_cfg.get("compression_ratio_threshold", 2.4),
            min_peak=asr_cfg.get("min_peak", DEFAULT_MIN_PEAK),
            filter_hallucinations=asr_cfg.get("filter_hallucinations", True),
            english_retry=asr_cfg.get("english_retry", False),
            cpu_model=asr_cfg.get("cpu_model"),
            routing=asr_cfg.get("routing", "sanskrit_aware"),
            en_confidence=asr_cfg.get("en_confidence", DEFAULT_EN_CONFIDENCE),
            sa_confidence=asr_cfg.get("sa_confidence", DEFAULT_SA_CONFIDENCE),
            sa_margin=asr_cfg.get("sa_margin", DEFAULT_SA_MARGIN),
        )
        kw.update(overrides)
        return cls(**kw)

    def resolved_backend(self) -> str:
        """mlx on Apple Silicon when available; openai-whisper otherwise."""
        if self._resolved_backend:
            return self._resolved_backend
        if self._backend == "mlx":
            self._resolved_backend = "mlx"
        elif self._backend in ("openai", "openai-whisper"):
            self._resolved_backend = "openai"
        else:
            try:
                import mlx_whisper  # noqa: F401
                self._resolved_backend = "mlx"
            except ImportError:
                self._resolved_backend = "openai"
        return self._resolved_backend

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
        duration_s = stats["duration_s"]

        if self._routing == "sanskrit_aware":
            text, lang = self._route(work, duration_s)
            # The tag follows the script actually produced: a forced-sa decode
            # that Whisper still wrote in Latin is English speech.
            if text and lang in ("sa", "en"):
                lang = language_detector.detect(text, lang)
        else:
            text, lang = self._legacy_auto(work, duration_s)

        if self._sanskrit_model and text and lang == "sa":
            try:
                text = self._transcribe_sanskrit(work)
                self.last_diag["sanskrit_model"] = True
            except Exception:  # experimental path must not break the turn (R7)
                logger.exception("asr: Sanskrit fine-tune failed; keeping Whisper sa decode")

        self.last_diag["transcript"] = text
        self.last_diag["asr_hint"] = lang
        self.last_diag["backend"] = self.resolved_backend()
        return text, lang

    # ------------------------------------------------------------- routing

    def _clean(self, result: dict, language: str | None, duration_s: float) -> str:
        """Decode text, or "" if it is a hallucination or the prompt echoed back."""
        text = (result.get("text") or "").strip()
        if not text or not self._filter_hallucinations:
            return text
        if is_hallucination(text, duration_s=duration_s):
            self.last_diag["hallucination"] = True
            logger.info("asr: dropped hallucination %r (%.2fs audio)", text, duration_s)
            return ""
        if is_prompt_leak(text, self._prompt_for(language)):
            self.last_diag["prompt_leak"] = True
            logger.info("asr: dropped prompt echo %r", text)
            return ""
        return text

    def _route(self, work: np.ndarray, duration_s: float) -> tuple[str, str | None]:
        scores = self.language_scores(work)
        self.last_diag["lid"] = scores
        en_p = scores["en"] if scores else 0.5
        sa_p = scores["sa"] if scores else 0.5

        if scores and en_p >= self._en_confidence:
            self.last_diag["route"] = "en"
            text = self._clean(self._decode(work, "en"), "en", duration_s)
            if not looks_unusable(text):
                return text, "en"
            # English was expected but nothing usable came back — the LID was
            # fooled (accented English and Sanskrit share a lot of phonetics).
            self.last_diag["route"] = "en→sa"
            return self._clean(self._decode(work, "sa"), "sa", duration_s), "sa"

        sa_res = self._decode(work, "sa")
        sa_text = self._clean(sa_res, "sa", duration_s)
        if scores and sa_p >= self._sa_confidence:
            self.last_diag["route"] = "sa"
            return sa_text, "sa"

        en_res = self._decode(work, "en")
        en_text = self._clean(en_res, "en", duration_s)
        sa_lp, en_lp = self._mean_logprob(sa_res), self._mean_logprob(en_res)
        self.last_diag["logprob"] = {"sa": round(sa_lp, 3), "en": round(en_lp, 3)}
        self.last_diag["route"] = "dual"
        if looks_unusable(en_text) and not looks_unusable(sa_text):
            return sa_text, "sa"
        if looks_unusable(sa_text) and not looks_unusable(en_text):
            return en_text, "en"
        if en_lp > sa_lp + self._sa_margin:
            return en_text, "en"
        return sa_text, "sa"

    def _legacy_auto(self, work: np.ndarray, duration_s: float) -> tuple[str, str | None]:
        """Pre-routing behaviour: free LID, then an optional English retry."""
        result = self._decode(work, language=None)
        lang = result.get("language")
        text = self._clean(result, None, duration_s)
        if self._english_retry and looks_unusable(text):
            retry = self._decode(work, language="en")
            retry_text = self._clean(retry, "en", duration_s)
            if retry_text:
                self.last_diag["english_retry"] = True
                logger.info("asr: english retry recovered %r (was %r)", retry_text, text)
                text, lang = retry_text, retry.get("language") or "en"
        return text, lang

    @staticmethod
    def _mean_logprob(result: dict) -> float:
        segs = [s for s in (result.get("segments") or []) if "avg_logprob" in s]
        if not segs:
            return -math.inf
        return float(sum(s["avg_logprob"] for s in segs) / len(segs))

    def language_scores(self, audio: np.ndarray) -> dict | None:
        """Whisper language-ID restricted to Mitra's three languages.

        Returns {"en", "sa", "top"} where "sa" pools every non-English
        language Whisper considered — on this robot, non-English speech is
        Sanskrit, whatever Whisper's free LID calls it (hi, ur, ar, vi, ...).
        "top" is Whisper's own unrestricted guess, kept for diagnostics.
        """
        try:
            probs = (self._lid_openai(audio) if self.resolved_backend() == "openai"
                     else self._lid_mlx(audio))
        except Exception:
            logger.exception("asr: language-ID failed; falling back to dual decode")
            return None
        if not probs:
            return None
        en = float(probs.get("en", 0.0))
        top = max(probs, key=probs.get)
        return {"en": round(en, 3), "sa": round(max(0.0, 1.0 - en), 3), "top": top}

    def _lid_mlx(self, audio: np.ndarray) -> dict:
        import mlx.core as mx
        from mlx_whisper.audio import N_FRAMES, N_SAMPLES, log_mel_spectrogram, pad_or_trim
        from mlx_whisper.transcribe import ModelHolder

        model = ModelHolder.get_model(self._default_model, mx.float16)
        if not model.is_multilingual:
            return {"en": 1.0}
        mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels, padding=N_SAMPLES)
        segment = pad_or_trim(mel, N_FRAMES, axis=-2).astype(mx.float16)
        _, probs = model.detect_language(segment)
        return dict(probs)

    def _lid_openai(self, audio: np.ndarray) -> dict:
        import whisper

        if self._openai_model is None:
            self._openai_model = whisper.load_model(self._openai_model_name())
        model = self._openai_model
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(audio),
                                          n_mels=model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        return dict(probs)

    def _prompt_for(self, language: str | None) -> str | None:
        if language == "sa":
            return self._sanskrit_prompt
        return self._initial_prompt

    # -------------------------------------------------------------- decode

    def _decode(self, audio: np.ndarray, language: str | None) -> dict:
        backend = self.resolved_backend()
        if backend == "openai":
            return self._decode_openai(audio, language)
        return self._decode_mlx(audio, language)

    def _decode_mlx(self, audio: np.ndarray, language: str | None) -> dict:
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
        prompt = self._prompt_for(language)
        if prompt:
            kwargs["initial_prompt"] = prompt
        try:
            return mlx_whisper.transcribe(audio, **kwargs)
        except TypeError:
            # Older mlx-whisper builds reject some decoding kwargs
            fallback = {"path_or_hf_repo": self._default_model}
            if language:
                fallback["language"] = language
            return mlx_whisper.transcribe(audio, **fallback)

    def _openai_model_name(self) -> str:
        if self._cpu_model:
            return self._cpu_model
        name = (self._default_model or "").lower()
        for size in ("large-v3", "large-v2", "turbo", "tiny", "base", "small", "medium", "large"):
            if size in name:
                return "turbo" if size == "turbo" else size
        return "small"

    def _decode_openai(self, audio: np.ndarray, language: str | None) -> dict:
        """CPU/Linux path so eval scripts share Transcriber with the app."""
        try:
            import whisper
        except ImportError as e:
            raise ImportError(
                "openai-whisper is required when mlx-whisper is unavailable. "
                "Install with: pip install 'mitra[asr-cpu]'"
            ) from e
        if self._openai_model is None:
            self._openai_model = whisper.load_model(self._openai_model_name())
        kwargs = {
            "condition_on_previous_text": self._condition_on_previous_text,
            "no_speech_threshold": self._no_speech_threshold,
            "compression_ratio_threshold": self._compression_ratio_threshold,
            "logprob_threshold": self._logprob_threshold,
        }
        if language:
            kwargs["language"] = language
        prompt = self._prompt_for(language)
        if prompt:
            kwargs["initial_prompt"] = prompt
        try:
            result = self._openai_model.transcribe(audio, **kwargs)
        except TypeError:
            result = self._openai_model.transcribe(
                audio, language=language or None
            )
        return {
            "text": (result.get("text") or "").strip(),
            "language": result.get("language"),
            "segments": result.get("segments") or [],
        }
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
        # the reliable path for our en/sa mix anyway.
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
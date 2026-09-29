"""Sanskrit TTS (FR-4.3): AI4Bharat Indic Parler-TTS, with a VITS path.

Two engines, selected by ``models.tts.engine`` in config.yaml:

- ``indic-parler-tts`` (mac / v1 target): the real Sanskrit voice, steered by
  a written ``voice_description`` rather than a named voice preset. A **gated**
  HuggingFace repo with auto-approval: open
  https://huggingface.co/ai4bharat/indic-parler-tts while logged in, accept the
  conditions, and (with `hf auth login` done) the model downloads normally. It
  also needs the parler-tts package, which is not on PyPI:

      pip install git+https://github.com/huggingface/parler-tts.git

  If either is missing, synthesize() falls back to VITS automatically.

- ``vits``: skips parler entirely. Required on Linux, where parler-tts pins
  transformers<=4.46.1 and so cannot be co-installed with reachy-mini (which
  needs huggingface-hub>=1.17). Uses facebook/mms-tts-hin, an ungated Hindi
  voice that reads Devanagari. Its Sanskrit pronunciation is approximate
  (visarga, vocalic ṛ), acceptable as a stopgap only (REQUIREMENTS R3; the
  design's quality fallback remains AI4Bharat Indic-TTS VITS). VITS has no
  voice-description steering, so ``voice_description`` is ignored here.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("mitra")

DEFAULT_VOICE = (
    "A female speaker delivers slow, warm, clear speech in a close-sounding "
    "recording with no background noise."
)
FALLBACK_MODEL = "facebook/mms-tts-hin"


class SanskritTTS:
    def __init__(self, model: str = "ai4bharat/indic-parler-tts",
                 device: str = "mps", voice_description: str = DEFAULT_VOICE,
                 fallback_model: str = FALLBACK_MODEL,
                 engine: str = "indic-parler-tts"):
        if engine not in ("indic-parler-tts", "vits"):
            raise ValueError(f"unknown TTS engine: {engine!r}")
        self._engine = engine
        self._model_id = model
        self._device = device
        self._voice = voice_description
        # engine: vits means models.tts.model IS the VITS voice, so honour it
        # over the fallback key (which exists for the parler-failure path).
        self._fallback_model = model if engine == "vits" else fallback_model
        self._parler = None   # ~2 GB — loaded on first synthesize()
        self._vits = None

    # ------------------------------------------------------------- loading

    def _ensure_loaded(self) -> None:
        if self._parler is not None or self._vits is not None:
            return
        if self._engine == "vits":
            self._load_vits()
            logger.info("TTS: VITS loaded (%s)", self._fallback_model)
            return
        try:
            self._load_parler()
            logger.info("TTS: Indic Parler-TTS loaded (%s)", self._model_id)
        except Exception as e:
            logger.warning(
                "TTS: %s unavailable (%s: %s) — falling back to %s. "
                "For the real Sanskrit voice: accept the conditions at "
                "https://huggingface.co/%s (auto-approved) and install "
                "parler-tts (see src/speech/tts.py docstring).",
                self._model_id, type(e).__name__, str(e)[:120],
                self._fallback_model, self._model_id,
            )
            self._load_vits()

    def _load_parler(self) -> None:
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self._torch = torch
        self._parler = ParlerTTSForConditionalGeneration.from_pretrained(
            self._model_id
        ).to(self._device)
        self._prompt_tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        desc_tokenizer = AutoTokenizer.from_pretrained(
            self._parler.config.text_encoder._name_or_path
        )
        self._voice_ids = desc_tokenizer(
            self._voice, return_tensors="pt"
        ).to(self._device)

    def _load_vits(self) -> None:
        import torch
        from transformers import AutoTokenizer, VitsModel

        self._torch = torch
        try:
            self._vits = VitsModel.from_pretrained(self._fallback_model).to(self._device)
            self._vits_device = self._device
        except Exception:  # MPS quirks / no CUDA → CPU is fine for a small VITS
            self._vits = VitsModel.from_pretrained(self._fallback_model)
            self._vits_device = "cpu"
        self._vits_tokenizer = AutoTokenizer.from_pretrained(self._fallback_model)

    # ---------------------------------------------------------- synthesis

    def synthesize(self, text_devanagari: str) -> tuple[np.ndarray, int]:
        """Devanagari text → (mono float32 waveform, samplerate)."""
        self._ensure_loaded()
        if self._parler is not None:
            return self._synthesize_parler(text_devanagari)
        return self._synthesize_vits(text_devanagari)

    def _synthesize_parler(self, text: str) -> tuple[np.ndarray, int]:
        prompt = self._prompt_tokenizer(text, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            generation = self._parler.generate(
                input_ids=self._voice_ids.input_ids,
                attention_mask=self._voice_ids.attention_mask,
                prompt_input_ids=prompt.input_ids,
                prompt_attention_mask=prompt.attention_mask,
            )
        wav = generation.cpu().numpy().squeeze().astype(np.float32)
        return wav, int(self._parler.config.sampling_rate)

    def _synthesize_vits(self, text: str) -> tuple[np.ndarray, int]:
        inputs = self._vits_tokenizer(text, return_tensors="pt").to(self._vits_device)
        with self._torch.no_grad():
            wav = self._vits(**inputs).waveform
        wav = wav.squeeze().cpu().numpy().astype(np.float32)
        return wav, int(self._vits.config.sampling_rate)
"""Utterance segmentation (FR-4.1): Silero VAD primary, energy fallback.

Both segmenters consume 16 kHz mono float32 chunks via ``process()`` and
return the complete utterance array once end-of-speech is detected, else None.
The energy segmenter is dependency-free and doubles as the test/dev fallback.
"""

from __future__ import annotations

import logging

import numpy as np

from . import TARGET_SAMPLERATE

logger = logging.getLogger("mitra")


class EnergySegmenter:
    """RMS-threshold segmentation with a silence hangover.

    By default the speech gate is adaptive: an EMA of the ambient noise floor,
    with speech = 3x the floor (bounded below by ``min_gate``). Mic capture
    levels vary hugely between machines/settings — a fixed threshold that works
    on one setup is deaf on another. Pass ``threshold`` to pin it instead.
    """

    _FLOOR_RATIO = 2.5

    def __init__(self, samplerate: int = TARGET_SAMPLERATE,
                 threshold: float | None = None, min_gate: float = 0.004,
                 min_speech_s: float = 0.3, min_silence_s: float = 0.8,
                 max_utterance_s: float = 15.0):
        self._sr = samplerate
        self._threshold = threshold
        self._min_gate = min_gate
        self._min_speech = min_speech_s * samplerate
        self._min_silence = min_silence_s * samplerate
        self._max_utterance = max_utterance_s * samplerate
        self._floor = min_gate / self._FLOOR_RATIO
        self.reset()

    def reset(self) -> None:
        self._buf: list[np.ndarray] = []
        self._speech_samples = 0
        self._silence_samples = 0
        self._in_speech = False

    def process(self, chunk: np.ndarray) -> np.ndarray | None:
        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if len(chunk) == 0:
            return None
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if self._threshold is not None:
            gate = self._threshold
        else:
            gate = max(self._FLOOR_RATIO * self._floor, self._min_gate)
        loud = rms >= gate
        if not loud and not self._in_speech:  # track ambient level while idle
            self._floor = 0.95 * self._floor + 0.05 * rms

        if not self._in_speech:
            if loud:
                self._in_speech = True
                self._buf = [chunk]
                self._speech_samples = len(chunk)
                self._silence_samples = 0
            return None

        self._buf.append(chunk)
        if loud:
            self._speech_samples += len(chunk)
            self._silence_samples = 0
        else:
            self._silence_samples += len(chunk)

        total = sum(len(c) for c in self._buf)
        ended = self._silence_samples >= self._min_silence or total >= self._max_utterance
        if not ended:
            return None
        utterance = np.concatenate(self._buf)
        enough_speech = self._speech_samples >= self._min_speech
        self.reset()
        return utterance if enough_speech else None


class SileroSegmenter:
    """Streaming Silero VAD via VADIterator (512-sample windows at 16 kHz)."""

    _WINDOW = 512

    def __init__(self, samplerate: int = TARGET_SAMPLERATE,
                 min_silence_s: float = 0.8, max_utterance_s: float = 15.0,
                 preroll_s: float = 0.2):
        try:
            import torch  # noqa: F401

            # silero_vad/model.py runs `torch.set_num_threads(1)` at import —
            # a *process-wide* setting, sensible for a 512-sample VAD window and
            # ruinous for the Whisper that shares this interpreter: single
            # threaded, whisper-small takes ~8 s on an utterance it transcribes
            # in ~1.5 s across cores. Restore whatever torch was set to before.
            threads = torch.get_num_threads()
            from silero_vad import VADIterator, load_silero_vad
        except ImportError as e:
            raise ImportError(
                "silero-vad (and torch) are required for the silero VAD engine. "
                "Install with: pip install 'mitra[vad]'"
            ) from e
        self._torch = __import__("torch")
        self._sr = samplerate
        self._iterator = VADIterator(
            load_silero_vad(), sampling_rate=samplerate,
            min_silence_duration_ms=int(min_silence_s * 1000),
        )
        # Restore after loading too, not just after the import: whichever call
        # clamps the pool, Whisper is the one that pays for it.
        if self._torch.get_num_threads() != threads:
            logger.info("silero-vad clamped torch to %d thread(s); restoring %d "
                        "— ASR shares this interpreter",
                        self._torch.get_num_threads(), threads)
            self._torch.set_num_threads(threads)
        self._max_utterance = int(max_utterance_s * samplerate)
        self._preroll = int(preroll_s * samplerate)
        self.reset()

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)  # not yet a full window
        self._history = np.zeros(0, dtype=np.float32)  # processed samples
        self._start: int | None = None
        try:
            self._iterator.reset_states()
        except AttributeError:
            pass

    def process(self, chunk: np.ndarray) -> np.ndarray | None:
        self._pending = np.concatenate(
            [self._pending, np.asarray(chunk, dtype=np.float32).reshape(-1)]
        )
        result = None
        while len(self._pending) >= self._WINDOW and result is None:
            window, self._pending = (
                self._pending[:self._WINDOW], self._pending[self._WINDOW:]
            )
            result = self._process_window(window)
        return result

    def _process_window(self, window: np.ndarray) -> np.ndarray | None:
        offset = len(self._history)
        self._history = np.concatenate([self._history, window])
        event = self._iterator(self._torch.from_numpy(window))
        if event and "start" in event:
            self._start = max(0, offset - self._preroll)
        elif self._start is not None:
            too_long = len(self._history) - self._start >= self._max_utterance
            if (event and "end" in event) or too_long:
                utterance = self._history[self._start:]
                self.reset()
                return utterance
        elif len(self._history) > self._max_utterance:
            self._history = self._history[-self._preroll:]  # cap idle memory
        return None


def make_segmenter(engine: str = "silero", **kwargs):
    """Build the configured segmenter, falling back to energy if silero is absent."""
    if engine == "silero":
        try:
            kwargs.pop("threshold", None)
            kwargs.pop("min_speech_s", None)
            return SileroSegmenter(**kwargs)
        except ImportError:
            logger.warning("silero-vad not installed; falling back to energy VAD")
            return EnergySegmenter()
    if engine == "energy":
        return EnergySegmenter(**kwargs)
    raise ValueError(f"unknown VAD engine: {engine!r}")

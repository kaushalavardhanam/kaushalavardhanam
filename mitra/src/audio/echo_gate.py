"""Playback-aware microphone gating (issue #9 self-echo).

Built-in-mic Mode A has no hardware AEC. During TTS playback the wake
transcriber hears Mitra's own voice (the 2026-09-16 log recorded six such
wake-check transcripts). This gate:

* suppresses wake *and* VAD while playback is active, plus a measured tail
* still allows barge-in when chunk RMS looks like a live talker, not speaker bleed

Gating and barge-in pull against each other; the energy threshold is the
compromise. Domain barge-in (``Event('wake')``) is unchanged.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

DEFAULT_TAIL_S = 0.45
DEFAULT_BARGE_IN_RMS = 0.08


def chunk_rms(samples: Any) -> float:
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


class EchoGate:
    """Drop self-echo; pass high-energy barge-in."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        playback_tail_s: float = DEFAULT_TAIL_S,
        barge_in_rms: float = DEFAULT_BARGE_IN_RMS,
    ):
        self.enabled = enabled
        self.playback_tail_s = float(playback_tail_s)
        self.barge_in_rms = float(barge_in_rms)
        self._playing = False
        self._tail_until = 0.0
        self.suppressed_chunks = 0
        self.barge_in_chunks = 0

    def notify_playback_started(self, duration_s: float | None = None) -> None:
        self._playing = True
        # Hold the gate at least until notify_playback_ended; duration is advisory.
        if duration_s:
            self._tail_until = max(
                self._tail_until,
                time.monotonic() + float(duration_s) + self.playback_tail_s,
            )

    def notify_playback_ended(self) -> None:
        self._playing = False
        self._tail_until = time.monotonic() + self.playback_tail_s

    def is_gated(self) -> bool:
        if not self.enabled:
            return False
        return self._playing or time.monotonic() < self._tail_until

    def allow(self, chunk) -> bool:
        """True if this chunk may be given to wake or VAD."""
        if not self.is_gated():
            return True
        rms = chunk_rms(chunk)
        if rms >= self.barge_in_rms:
            self.barge_in_chunks += 1
            return True
        self.suppressed_chunks += 1
        return False

    allow_wake = allow
    allow_listen = allow

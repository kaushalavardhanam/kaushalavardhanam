"""Lightweight frames for the Pipecat proof of concept.

When the Daily ``pipecat`` package is installed these map onto its Frame
types; otherwise they are standalone dataclasses so the PoC stays testable
without that dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


try:  # optional — never required to import mitra
    from pipecat.frames.frames import (  # type: ignore
        Frame as _PipeFrame,
        InputAudioRawFrame,
        TranscriptionFrame,
        TextFrame,
        LLMMessagesFrame,
        TTSAudioRawFrame,
        InterruptionFrame,
        StartFrame,
        EndFrame,
    )
    PIPECAT_SDK = True
except ImportError:
    PIPECAT_SDK = False
    _PipeFrame = object  # type: ignore
    InputAudioRawFrame = TranscriptionFrame = TextFrame = object  # type: ignore
    LLMMessagesFrame = TTSAudioRawFrame = InterruptionFrame = object  # type: ignore
    StartFrame = EndFrame = object  # type: ignore


@dataclass
class MitraFrame:
    """One datum moving through the Mitra/Pipecat processor chain."""

    kind: str
    payload: Any = None
    meta: dict = field(default_factory=dict)


# Kind constants used by processors and tests
AUDIO = "audio"
WAKE = "wake"
UTTERANCE = "utterance"
TRANSCRIPT = "transcript"
REPLY = "reply"
INTERRUPT = "interrupt"
PLAYBACK_DONE = "playback_done"
STOP = "stop"

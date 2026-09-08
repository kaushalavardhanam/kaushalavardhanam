"""Pipecat proof of concept (issue #7).

Selectable with ``orchestration.engine: pipecat`` or ``--orchestrator pipecat``.
The existing custom orchestrator remains the default.

What this path replaces
-----------------------
Only the *scheduler*: audio chunks and turn messages move through a
Frame-processor pipeline (Pipecat's architectural unit) instead of the
hand-written ``_audio_loop`` / ``handle_event`` dispatch.

What stays Mitra-specific (not replaced)
----------------------------------------
Wake matching, Silero/energy VAD, local Whisper, language tags, Strands
tools, Devanagari validation, verified lexicon, Indic TTS, barge-in, and
``flush_mic`` after playback. Those are domain requirements Pipecat should
not rewrite.

Daily's full Pipecat stack assumes WebRTC transports and async services for
cloud STT/TTS. That would send raw microphone audio off-host and bypass
Mitra's privacy boundary, so this PoC uses Pipecat's *processor model* with
a custom Reachy transport. When the ``pipecat`` package is absent, a
compatible shim keeps the path testable.

Recommendation lives in ``evals/ADR-001-pipecat-bedrock.md``.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable

from mitra.audio import TARGET_SAMPLERATE, resample
from mitra.orchestration.frames import (
    AUDIO,
    INTERRUPT,
    MitraFrame,
    PIPECAT_SDK,
    PLAYBACK_DONE,
    REPLY,
    STOP,
    TRANSCRIPT,
    UTTERANCE,
    WAKE,
)
from mitra.orchestrator import Event, Orchestrator, State

logger = logging.getLogger("mitra")


class Processor:
    """Minimal FrameProcessor: process one frame, emit zero or more."""

    name = "processor"

    def process(self, frame: MitraFrame) -> list[MitraFrame]:
        return [frame]


class Pipeline:
    def __init__(self, processors: Iterable[Processor], name: str = "mitra"):
        self.processors = list(processors)
        self.name = name
        self.sdk = "pipecat" if PIPECAT_SDK else "mitra-shim"

    def push(self, frame: MitraFrame) -> list[MitraFrame]:
        batch = [frame]
        for proc in self.processors:
            nxt: list[MitraFrame] = []
            for item in batch:
                nxt.extend(proc.process(item))
            batch = nxt
        return batch


class ResampleProcessor(Processor):
    name = "resample"

    def __init__(self, src_rate: int, dst_rate: int = TARGET_SAMPLERATE):
        self.src_rate = src_rate
        self.dst_rate = dst_rate

    def process(self, frame: MitraFrame) -> list[MitraFrame]:
        if frame.kind != AUDIO or self.src_rate == self.dst_rate:
            return [frame]
        frame.payload = resample(frame.payload, self.src_rate, self.dst_rate)
        frame.meta["samplerate"] = self.dst_rate
        return [frame]


class WakeGateProcessor(Processor):
    name = "wake"

    def __init__(self, wake, state_fn: Callable[[], State]):
        self.wake = wake
        self.state_fn = state_fn

    def process(self, frame: MitraFrame) -> list[MitraFrame]:
        if frame.kind != AUDIO or self.wake is None:
            return [frame]
        state = self.state_fn()
        if state in (State.ASLEEP, State.SPEAKING, State.WAKING):
            if self.wake.process(frame.payload):
                return [MitraFrame(WAKE, meta=dict(frame.meta))]
            return []
        return [frame]


class VadProcessor(Processor):
    name = "vad"

    def __init__(self, segmenter, state_fn: Callable[[], State]):
        self.segmenter = segmenter
        self.state_fn = state_fn

    def process(self, frame: MitraFrame) -> list[MitraFrame]:
        if frame.kind != AUDIO or self.segmenter is None:
            return [frame]
        if self.state_fn() != State.LISTENING:
            return []
        utterance = self.segmenter.process(frame.payload)
        if utterance is None:
            return []
        return [MitraFrame(UTTERANCE, utterance, dict(frame.meta))]


class AsrProcessor(Processor):
    name = "asr"

    def __init__(self, asr):
        self.asr = asr

    def process(self, frame: MitraFrame) -> list[MitraFrame]:
        if frame.kind != UTTERANCE:
            return [frame]
        if isinstance(frame.payload, str):
            return [MitraFrame(TRANSCRIPT, (frame.payload, None), dict(frame.meta))]
        text, hint = self.asr.transcribe(frame.payload)
        meta = dict(frame.meta)
        if getattr(self.asr, "last_diag", None):
            meta["asr_diag"] = self.asr.last_diag
        return [MitraFrame(TRANSCRIPT, (text, hint), meta)]


class MitraTurnProcessor(Processor):
    """Hands a transcript to the existing Orchestrator turn logic.

    Validation, lexicon, tools, and TTS stay deterministic and unchanged.
    """

    name = "mitra_turn"

    def __init__(self, orchestrator: Orchestrator):
        self.orchestrator = orchestrator

    def process(self, frame: MitraFrame) -> list[MitraFrame]:
        if frame.kind == WAKE:
            self.orchestrator.handle_event(Event("wake"))
            return [frame]
        if frame.kind == UTTERANCE:
            self.orchestrator.handle_event(Event("utterance", frame.payload))
            return [MitraFrame(REPLY, meta=dict(frame.meta))]
        if frame.kind == TRANSCRIPT:
            text, _hint = frame.payload
            self.orchestrator.handle_event(Event("utterance", text))
            return [MitraFrame(REPLY, text, dict(frame.meta))]
        if frame.kind == INTERRUPT:
            self.orchestrator.handle_event(Event("wake"))
            return [frame]
        if frame.kind == PLAYBACK_DONE:
            self.orchestrator.handle_event(Event("playback_done"))
            return [frame]
        if frame.kind == STOP:
            self.orchestrator.handle_event(Event("stop"))
            return [frame]
        return [frame]


class PipecatOrchestrator(Orchestrator):
    """Drop-in Orchestrator that pumps mic audio through a Frame pipeline.

    ``handle_event`` is unchanged so barge-in, silence timeout, validation,
    lexicon, and session reset keep the existing tests and behaviour.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        src_rate = getattr(self.robot, "mic_samplerate", TARGET_SAMPLERATE)
        self.pipeline = Pipeline(
            [
                ResampleProcessor(src_rate, TARGET_SAMPLERATE),
                WakeGateProcessor(self.wake, lambda: self.state),
                VadProcessor(self.segmenter, lambda: self.state),
                MitraTurnProcessor(self),
            ],
            name="mitra-pipecat",
        )
        logger.info("pipecat PoC pipeline ready (sdk=%s): %s",
                    self.pipeline.sdk,
                    " → ".join(p.name for p in self.pipeline.processors))

    def _audio_loop(self) -> None:
        """Pump mic chunks through the frame pipeline instead of ad-hoc ifs."""
        while not self._stop.is_set():
            try:
                chunk = self.robot.mic_read()
            except Exception:
                self.logger.exception("microphone read failure")
                time_sleep_retry()
                continue
            if chunk is None or len(chunk) == 0:
                continue
            self.pipeline.push(MitraFrame(AUDIO, chunk, {"samplerate": self.robot.mic_samplerate}))


def time_sleep_retry() -> None:
    import time
    time.sleep(0.5)


def components_replaced() -> dict[str, str]:
    return {
        "audio pump / state dispatch": "Pipecat-style processor pipeline",
        "wake / VAD / ASR / LLM / TTS / validator / lexicon": "unchanged Mitra modules",
        "Daily WebRTC transport": "not used (would leave the privacy boundary)",
        "Pipecat cloud STT/TTS services": "not used (raw audio must stay on host)",
    }

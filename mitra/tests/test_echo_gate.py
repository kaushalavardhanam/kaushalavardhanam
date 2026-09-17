"""Self-echo must not reach the wake transcriber; barge-in still works."""

import numpy as np

from mitra.audio.echo_gate import EchoGate
from mitra.orchestration.frames import AUDIO, MitraFrame, WAKE
from mitra.orchestration.pipecat_runtime import Pipeline, WakeGateProcessor
from mitra.orchestrator import Event, State


class RecordingWake:
    def __init__(self, fire: bool = False):
        self.heard: list = []
        self.fire = fire

    def process(self, chunk):
        # Stand-in for the wake ASR that transcribed Mitra's own playback.
        self.heard.append("Welcome, Krodipa.")
        return self.fire

    def reset(self):
        pass


def _bleed(n=1600, amp=0.01):
    return np.full(n, amp, dtype=np.float32)


def _barge(n=1600, amp=0.3):
    return np.full(n, amp, dtype=np.float32)


def test_echo_gate_drops_low_rms_during_playback():
    gate = EchoGate(playback_tail_s=0.4, barge_in_rms=0.08)
    gate.notify_playback_started(duration_s=1.0)
    assert gate.allow_wake(_bleed()) is False
    assert gate.suppressed_chunks == 1


def test_echo_gate_passes_barge_in_energy():
    gate = EchoGate(barge_in_rms=0.08)
    gate.notify_playback_started()
    assert gate.allow_wake(_barge()) is True
    assert gate.barge_in_chunks == 1


def test_echo_gate_inactive_when_idle():
    gate = EchoGate()
    assert gate.allow_wake(_bleed()) is True


def test_custom_path_does_not_transcribe_playback(make_orchestrator):
    wake = RecordingWake(fire=False)
    orch, _ = make_orchestrator(wake=wake)
    orch.state = State.SPEAKING
    orch.echo_gate.notify_playback_started(duration_s=2.0)
    # Reproduce the audio-loop decision without starting threads.
    chunk = _bleed()
    if orch.echo_gate.allow_wake(chunk):
        orch.wake.process(chunk)
    assert wake.heard == []
    assert orch.echo_gate.suppressed_chunks >= 1


def test_custom_path_barge_in_still_interrupts(make_orchestrator, fake_robot):
    wake = RecordingWake(fire=True)
    orch, _ = make_orchestrator(wake=wake)
    fake_robot.hold_playback = True
    orch.state = State.SPEAKING
    orch.echo_gate.notify_playback_started()
    chunk = _barge()
    assert orch.echo_gate.allow_wake(chunk) is True
    if orch.wake.process(chunk):
        orch.handle_event(Event("wake"))
    assert wake.heard == ["Welcome, Krodipa."]
    assert fake_robot.stops == 1
    assert orch.state == State.LISTENING


def test_pipecat_wake_gate_suppresses_echo():
    wake = RecordingWake(fire=True)
    gate = EchoGate(barge_in_rms=0.08)
    gate.notify_playback_started()
    pipe = Pipeline([
        WakeGateProcessor(wake, lambda: State.SPEAKING, echo_gate=gate),
    ])
    out = pipe.push(MitraFrame(AUDIO, _bleed()))
    assert out == []
    assert wake.heard == []


def test_pipecat_wake_gate_allows_barge_in():
    wake = RecordingWake(fire=True)
    gate = EchoGate(barge_in_rms=0.08)
    gate.notify_playback_started()
    pipe = Pipeline([
        WakeGateProcessor(wake, lambda: State.SPEAKING, echo_gate=gate),
    ])
    out = pipe.push(MitraFrame(AUDIO, _barge()))
    assert out and out[0].kind == WAKE
    assert wake.heard == ["Welcome, Krodipa."]


def test_handle_event_wake_still_barges_on_pipecat(make_orchestrator, fake_robot):
    from mitra.orchestration.pipecat_runtime import PipecatOrchestrator

    orch, _ = make_orchestrator()
    pipe = PipecatOrchestrator(
        robot=orch.robot, agent=orch.agent, tts=orch.tts, lexicon=orch.lexicon,
        wake=RecordingWake(),
    )
    fake_robot.hold_playback = True
    pipe.state = State.SPEAKING
    pipe.handle_event(Event("wake"))
    assert fake_robot.stops == 1
    assert pipe.state == State.LISTENING

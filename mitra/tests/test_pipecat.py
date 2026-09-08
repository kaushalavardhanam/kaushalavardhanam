import numpy as np

from mitra.agent import prompts
from mitra.orchestration.frames import AUDIO, MitraFrame, UTTERANCE, WAKE
from mitra.orchestration.pipecat_runtime import (
    MitraTurnProcessor,
    Pipeline,
    PipecatOrchestrator,
    ResampleProcessor,
    VadProcessor,
    WakeGateProcessor,
    components_replaced,
)
from mitra.orchestrator import Event, State


def test_pipecat_drop_in_matches_custom_wake(make_orchestrator, fake_robot, fake_tts):
    orch, _ = make_orchestrator()
    pipe = PipecatOrchestrator(
        robot=orch.robot, agent=orch.agent, tts=orch.tts, lexicon=orch.lexicon,
    )
    pipe.handle_event(Event("wake"))
    assert pipe.state == State.WAKING
    assert fake_robot.nods == 1
    assert fake_tts.spoken == [prompts.GREETING]


def test_pipecat_text_turn_validates_and_speaks(make_orchestrator, fake_tts):
    orch, agent = make_orchestrator(replies=["मम नाम मित्रम्।"])
    pipe = PipecatOrchestrator(
        robot=orch.robot, agent=orch.agent, tts=orch.tts, lexicon=orch.lexicon,
    )
    pipe.state = State.LISTENING
    pipe.handle_event(Event("utterance", "What is your name?"))
    assert fake_tts.spoken == ["मम नाम मित्रम्।"]
    assert agent.calls[0].startswith("[lang=en]")


def test_pipecat_barge_in(make_orchestrator, fake_robot):
    orch, _ = make_orchestrator()
    pipe = PipecatOrchestrator(
        robot=orch.robot, agent=orch.agent, tts=orch.tts, lexicon=orch.lexicon,
    )
    fake_robot.hold_playback = True
    pipe.state = State.SPEAKING
    pipe.handle_event(Event("wake"))
    assert fake_robot.stops == 1
    assert pipe.state == State.LISTENING


def test_pipeline_wake_gate_emits_wake():
    class Wake:
        def process(self, _chunk):
            return True

    pipe = Pipeline([
        WakeGateProcessor(Wake(), lambda: State.ASLEEP),
        VadProcessor(None, lambda: State.ASLEEP),
    ])
    out = pipe.push(MitraFrame(AUDIO, np.zeros(160, dtype=np.float32)))
    assert out and out[0].kind == WAKE


def test_pipeline_vad_emits_utterance_when_listening():
    class Seg:
        def process(self, chunk):
            return chunk

    pipe = Pipeline([VadProcessor(Seg(), lambda: State.LISTENING)])
    chunk = np.ones(160, dtype=np.float32)
    out = pipe.push(MitraFrame(AUDIO, chunk))
    assert out[0].kind == UTTERANCE
    assert len(out[0].payload) == 160


def test_resample_processor_changes_rate():
    proc = ResampleProcessor(32000, 16000)
    frame = MitraFrame(AUDIO, np.zeros(320, dtype=np.float32))
    out = proc.process(frame)[0]
    assert len(out.payload) == 160


def test_turn_processor_routes_to_orchestrator(make_orchestrator, fake_tts):
    orch, _ = make_orchestrator(replies=["नमस्ते मित्र।"])
    orch.state = State.LISTENING
    MitraTurnProcessor(orch).process(MitraFrame(UTTERANCE, "hello"))
    assert fake_tts.spoken == ["नमस्ते मित्र।"]


def test_pipecat_audio_pipeline_does_not_call_handle_event(make_orchestrator):
    """Live path queues events; handle_event stays on the run-loop thread."""
    orch, _ = make_orchestrator()

    class Wake:
        def process(self, _chunk):
            return True

    pipe = PipecatOrchestrator(
        robot=orch.robot, agent=orch.agent, tts=orch.tts, lexicon=orch.lexicon,
        wake=Wake(),
    )
    assert all(p.name != "mitra_turn" for p in pipe.pipeline.processors)
    out = pipe.pipeline.push(MitraFrame(AUDIO, np.zeros(160, dtype=np.float32)))
    assert pipe.state == State.ASLEEP
    assert pipe.events.empty()
    pipe.enqueue_pipeline_output(out)
    ev = pipe.events.get_nowait()
    assert ev.kind == "wake"


def test_components_replaced_keeps_domain_logic():
    mapping = components_replaced()
    assert "unchanged Mitra modules" in mapping["wake / VAD / ASR / LLM / TTS / validator / lexicon"]
    assert "not used" in mapping["Daily WebRTC transport"].lower() or "not used" in mapping["Daily WebRTC transport"]


def test_pipecat_injects_all_ten_conversation_scenarios(make_orchestrator, fake_tts):
    """Post-ASR Mode A inject through Pipecat — same handle_event path as custom."""
    from mitra.eval.corpus import conversation_scenarios
    from mitra.eval.sanskrit_reference import REFERENCE_REPLIES

    scenarios = conversation_scenarios()
    replies = [REFERENCE_REPLIES[s["id"]]["sanskrit"] for s in scenarios]
    orch, agent = make_orchestrator(replies=replies)
    pipe = PipecatOrchestrator(
        robot=orch.robot, agent=orch.agent, tts=orch.tts, lexicon=orch.lexicon,
    )
    pipe.state = State.LISTENING
    for scenario in scenarios:
        pipe.handle_event(Event("utterance", scenario["expected"]))
        pipe.state = State.LISTENING
    assert fake_tts.spoken == replies
    assert all(c.startswith("[lang=en]") for c in agent.calls)

"""ttfa, ollama_loaded, and ASR first-error diagnostics (issue #9)."""

import time

import numpy as np

from mitra.audio.hallucinations import looks_unusable
from mitra.logging_subsystem import TurnLogger
from mitra.orchestrator import Event, State
from mitra.pipeline_trace import first_error_stage


class SlowTTS:
    def __init__(self, delay_s=0.05):
        self.spoken = []
        self.delay_s = delay_s

    def synthesize(self, text):
        time.sleep(self.delay_s)
        self.spoken.append(text)
        return np.zeros(1600, dtype=np.float32), 16000


class DelayedAgent:
    def __init__(self, reply, delay_s=0.04):
        self.reply = reply
        self.delay_s = delay_s
        self.last_usage = {}

    def converse(self, message):
        time.sleep(self.delay_s)
        return self.reply

    def reset(self):
        pass


def test_looks_unusable_leading_digit():
    assert looks_unusable("2 play sports")
    assert not looks_unusable("Do you play sports?")


def test_first_error_stage_do_to_two_is_asr():
    stage = first_error_stage({
        "payload_kind": "audio",
        "transcript": "2 play sports",
        "audio_stats": {"n_samples": 16000, "peak": 0.2},
        "reply": "अहं खेलनीयं न अस्मि। भवान् कथम्?",
        "validation_ok": True,
    })
    assert stage == "asr"


def test_first_error_stage_quality_is_not_script_validation():
    stage = first_error_stage({
        "payload_kind": "text",
        "transcript": "Where do you live?",
        "reply": "अहं तुम्हांस्केत्रे अस्मि।",
        "validation_ok": True,
        "quality_ok": False,
    })
    assert stage == "quality"


def test_ttfa_is_not_tts_synth_duration(lexicon, fake_robot, tmp_path):
    from mitra.orchestrator import Orchestrator

    tts = SlowTTS(0.05)
    agent = DelayedAgent("मम नाम मित्रम्।", delay_s=0.04)
    logger = TurnLogger(tmp_path)
    orch = Orchestrator(
        robot=fake_robot, agent=agent, tts=tts, lexicon=lexicon,
        turn_logger=logger,
        llm_meta={"provider": "ollama", "model_id": "qwen3-vl:8b-instruct",
                  "region": None, "ollama_loaded": True, "ollama_contacted": True},
    )
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "What is your name?"))
    record = logger.path.read_text(encoding="utf-8").strip()
    assert record
    import json
    row = json.loads(record)
    assert row["tts_synth_s"] >= 0.04
    # End of utterance → first audio includes LLM time; synthesis is separate.
    assert row["ttfa_s"] >= row["tts_synth_s"] + 0.03
    assert row["ollama_loaded"] is True
    assert row["region"] is None


def test_bedrock_turn_records_explicit_ollama_false(lexicon, fake_robot, tmp_path):
    from mitra.orchestrator import Orchestrator

    tts = SlowTTS(0.0)
    agent = DelayedAgent("मम नाम मित्रम्।", delay_s=0.0)
    agent.last_usage = {"input_tokens": 120, "output_tokens": 18}
    logger = TurnLogger(tmp_path)
    orch = Orchestrator(
        robot=fake_robot, agent=agent, tts=tts, lexicon=lexicon,
        turn_logger=logger,
        llm_meta={"provider": "bedrock", "model_id": "us.anthropic.claude-sonnet-4-6",
                  "region": "us-west-2", "ollama_loaded": False,
                  "ollama_contacted": False},
    )
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "What is your name?"))
    import json
    row = json.loads(logger.path.read_text(encoding="utf-8").strip())
    assert row["provider"] == "bedrock"
    assert row["region"] == "us-west-2"
    assert row["ollama_loaded"] is False
    assert row["ollama_contacted"] is False
    assert row["input_tokens"] == 120
    assert row["est_usd"] is not None


def test_quality_retry_does_not_flip_script_ok(make_orchestrator, fake_tts):
    bad = "अहं स्वागतम् करोति। भवान् कथम्?"
    good = "अहं त्वया सह वदामि।"
    orch, agent = make_orchestrator(replies=[bad, good])
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "What are you doing?"))
    assert len(agent.calls) == 2
    from mitra.agent import prompts
    assert agent.calls[1].endswith(prompts.QUALITY_CORRECTIVE_SUFFIX)
    assert fake_tts.spoken == [good]

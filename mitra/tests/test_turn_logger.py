from mitra.logging_subsystem import TurnLogger


def test_emit_records_first_error_stage(tmp_path):
    tl = TurnLogger(tmp_path)
    tl.start_turn()
    tl.set("payload_kind", "audio")
    tl.set("asr_hallucination", True)
    tl.set("transcript", "thanks for watching")
    tl.set("expected_transcript", "What are you doing?")
    tl.set("audio_stats", {"n_samples": 8000, "peak": 0.2})
    tl.set("provider", "bedrock")
    tl.set("model_id", "us.amazon.nova-pro-v1:0")
    record = tl.emit()
    assert record["first_error_stage"] == "asr"
    lines = (tmp_path / "turns.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_orchestrator_logs_provider(make_orchestrator, tmp_path):
    from mitra.logging_subsystem import TurnLogger
    from mitra.orchestrator import Event, State

    tl = TurnLogger(tmp_path)
    orch, _ = make_orchestrator(
        replies=["मम नाम मित्रम्।"],
        turn_logger=tl,
        llm_meta={"provider": "bedrock", "model_id": "us.amazon.nova-pro-v1:0",
                  "region": "us-west-2"},
    )
    orch.state = State.LISTENING
    orch.handle_event(Event("utterance", "What is your name?"))
    record = tl.path.read_text(encoding="utf-8")
    assert "nova-pro" in record
    assert "What is your name?" in record

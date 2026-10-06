from mitra.agent.tools import END_SESSION_SENTINEL, build_tools


def _by_name(tools):
    # Without strands installed the factory returns plain functions; with
    # strands, DecoratedFunctionTool still passes calls through to them.
    return {getattr(t, "tool_name", getattr(t, "__name__", str(t))): t for t in tools}


def test_tools_built_without_a_speech_tool(fake_robot, fake_tts):
    """Speech is the orchestrator's job; a model-callable speak tool played
    unvalidated text and double-spoke replies."""
    tools = _by_name(build_tools(fake_robot, fake_tts))
    assert set(tools) == {"capture_image", "nod", "end_session"}


def test_capture_image_returns_jpeg(fake_robot, fake_tts):
    result = _by_name(build_tools(fake_robot, fake_tts))["capture_image"]()
    assert result["format"] == "jpeg"
    assert result["source"]["bytes"][:2] == b"\xff\xd8"  # JPEG magic


def test_nod_moves_head(fake_robot, fake_tts):
    _by_name(build_tools(fake_robot, fake_tts))["nod"]()
    assert fake_robot.nods == 1


def test_end_session_sentinel(fake_robot, fake_tts):
    assert _by_name(build_tools(fake_robot, fake_tts))["end_session"]() == END_SESSION_SENTINEL

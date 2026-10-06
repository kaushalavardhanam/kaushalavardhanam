"""Conversation-history trimming (src/agent/agent.py).

Strands is an optional extra, so these drive the trimming logic against a stub
that mimics the provider message shape rather than constructing a real Agent.
"""

from __future__ import annotations

from mitra.agent.agent import MitraAgent


class _StubAgent:
    def __init__(self, messages):
        self.messages = messages


def _mk(n):
    """n user/assistant exchanges as plain text messages."""
    out = []
    for i in range(n):
        out.append({"role": "user", "content": [{"text": f"q{i}"}]})
        out.append({"role": "assistant", "content": [{"text": f"a{i}"}]})
    return out


def _trimmer(messages, turns=4):
    agent = MitraAgent.__new__(MitraAgent)
    agent.max_history_turns = turns
    agent._agent = _StubAgent(messages)
    return agent


def test_short_history_untouched():
    agent = _trimmer(_mk(3))
    agent._trim_history()
    assert len(agent._agent.messages) == 6


def test_long_history_trimmed_to_window():
    agent = _trimmer(_mk(10))
    agent._trim_history()
    assert len(agent._agent.messages) == 8


def test_trim_keeps_the_most_recent_turns():
    agent = _trimmer(_mk(10))
    agent._trim_history()
    assert agent._agent.messages[-1]["content"][0]["text"] == "a9"


def test_window_starts_on_a_user_turn():
    agent = _trimmer(_mk(10))
    agent._trim_history()
    assert agent._agent.messages[0]["role"] == "user"


def test_window_never_opens_on_an_orphaned_tool_result():
    """A toolResult without its toolUse is a malformed conversation."""
    messages = _mk(4)
    messages.insert(-1, {"role": "user",
                         "content": [{"toolResult": {"content": []}}]})
    agent = _trimmer(messages, turns=2)
    agent._trim_history()
    first = agent._agent.messages[0]
    assert first["role"] == "user"
    assert not any("toolResult" in b for b in first["content"])


def test_zero_disables_trimming():
    agent = _trimmer(_mk(10), turns=0)
    agent._trim_history()
    assert len(agent._agent.messages) == 20


def test_unexpected_message_shape_leaves_history_alone():
    agent = _trimmer("not a list")
    agent._trim_history()
    assert agent._agent.messages == "not a list"


def test_rollback_drops_the_failed_turn_including_retries():
    msgs = _mk(2)
    agent = _trimmer(msgs)
    mark = agent.history_mark()
    # A failed turn: question, bad reply, corrective retry, bad reply.
    agent._agent.messages = msgs + _mk(2)
    agent.rollback(mark)
    assert agent._agent.messages == _mk(2)


def test_rollback_from_empty_history_clears():
    agent = _trimmer([])
    mark = agent.history_mark()
    agent._agent.messages = _mk(1)
    agent.rollback(mark)
    assert agent._agent.messages == []


def test_converse_trims_history():
    """Regression: an early `return` made _trim_history unreachable."""
    agent = _trimmer(_mk(10), turns=2)
    agent.last_error = None
    agent.provider = agent.model_id = agent.region = None

    class _CallableStub(_StubAgent):
        def __call__(self, message):
            self.messages.append({"role": "user", "content": [{"text": message}]})
            self.messages.append({"role": "assistant", "content": [{"text": "r"}]})
            return "r"

    agent._agent = _CallableStub(agent._agent.messages)
    assert agent.converse("hello") == "r"
    assert len(agent._agent.messages) == 4
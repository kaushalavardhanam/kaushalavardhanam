"""Strands Agent construction (DESIGN §1.3–1.5).

Provider selection is config-driven: local Ollama (Qwen3-VL) by default;
bedrock/anthropic is the Option B swap (FR-6.3, issue #7). The Strands Agent
owns the conversation loop, message history, and tool dispatch.

Bedrock mode never constructs an Ollama model. Fallback is explicit and
opt-in via ``models.llm.fallback.enabled``.
"""

from __future__ import annotations

import logging

from . import provider as llm_provider
from .errors import ProviderError
from .prompts import SANSKRIT_SYSTEM_PROMPT

logger = logging.getLogger("mitra")
# Turns of history kept in the agent's context. The model imitates its own
# recent output more strongly than the few-shot examples: in one logged
# session turn 1 produced the correct भवान् कथम्?, then drifted to the
# ungrammatical कथं भवतः? and repeated it for five straight turns. A short
# window keeps the conversation coherent while letting a bad pattern age out
# instead of compounding. 0 disables trimming.
DEFAULT_MAX_HISTORY_TURNS = 4


class MitraAgent:
    def __init__(self, llm_config: dict, tools: list,
                 system_prompt: str = SANSKRIT_SYSTEM_PROMPT, verbose: bool = True,
                 max_history_turns: int = DEFAULT_MAX_HISTORY_TURNS):
        try:
            from strands import Agent
        except ImportError as e:
            raise ImportError(
                "strands-agents is required for the agent layer. "
                "Install with: pip install 'mitra[agent]'"
            ) from e
        self.llm_config = dict(llm_config)
        self.provider = llm_provider.provider_name(llm_config)
        self.model_id = llm_provider.model_id(llm_config)
        self.region = llm_provider.resolve_region(llm_config) if self.provider == "bedrock" else None
        self.last_error: ProviderError | None = None
        # Strands' default callback handler streams reply tokens straight to
        # stdout — set verbose=False (e.g. in batch/test scripts) to silence
        # it and get only the final string from converse().
        self.max_history_turns = max_history_turns
        agent_kwargs = {} if verbose else {"callback_handler": None}
        self._agent = Agent(
            model=llm_provider.make_model(llm_config),
            tools=tools,
            system_prompt=system_prompt,
            **agent_kwargs,
        )

    @staticmethod
    def _make_model(cfg: dict):
        """Back-compat wrapper — prefer ``mitra.agent.provider.make_model``."""
        return llm_provider.make_model(cfg)

    def converse(self, message: str) -> str:
        """One turn: user message in, final agent text out (tools may run)."""
        self.last_error = None
        try:
            reply = str(self._agent(message)).strip()
        except Exception as e:
            err = llm_provider.map_provider_exception(
                e, provider=self.provider, model_id=self.model_id, region=self.region,
            )
            self.last_error = err
            logger.error("LLM provider=%s model=%s failed: %s",
                         self.provider, self.model_id, err)
            raise err from e
        # Previously unreachable (merge leftover: an early `return` sat above
        # it), so history grew without bound and bad phrasings never aged out.
        self._trim_history()
        return reply

    def history_mark(self):
        """Opaque marker for 'history as of now' (see ``rollback``)."""
        messages = getattr(self._agent, "messages", None)
        if not isinstance(messages, list) or not messages:
            return None
        return messages[-1]

    def rollback(self, mark) -> None:
        """Drop every message added after ``mark`` — i.e. the whole failed turn,
        including any corrective retry. Matching is by identity because
        trimming may shift indices; a marker that was trimmed away means the
        turn filled the window, so clearing it all loses nothing older."""
        messages = getattr(self._agent, "messages", None)
        if not isinstance(messages, list):
            return
        if mark is None:
            self._agent.messages = []
            return
        for i in range(len(messages) - 1, -1, -1):
            if messages[i] is mark:
                self._agent.messages = messages[: i + 1]
                return
        self._agent.messages = []

    @staticmethod
    def _is_clean_start(message) -> bool:
        """True if history may begin here — a plain user turn.

        A window that opens on a tool result, or on the assistant's half of an
        exchange, is not a conversation the provider will accept.
        """
        if not isinstance(message, dict) or message.get("role") != "user":
            return False
        content = message.get("content")
        if isinstance(content, list):
            return not any(isinstance(block, dict) and "toolResult" in block
                           for block in content)
        return True

    def _trim_history(self) -> None:
        """Hold the context to the last ``max_history_turns`` exchanges.

        Defensive throughout: Strands owns this list and its shape is the
        provider's, not ours, so anything unexpected leaves history untouched
        rather than risking a malformed conversation.
        """
        if self.max_history_turns <= 0:
            return
        messages = getattr(self._agent, "messages", None)
        if not isinstance(messages, list):
            return
        keep = self.max_history_turns * 2
        if len(messages) <= keep:
            return
        window = messages[-keep:]
        while window and not self._is_clean_start(window[0]):
            window.pop(0)
        if window:
            self._agent.messages = window

    def reset(self) -> None:
        """Drop conversation history at session end (FR-3.3: per-session context)."""
        self._agent.messages = []


class ExplicitFallbackAgent:
    """Wraps a primary agent; uses a second provider only when configured.

    Used when ``models.llm.fallback.enabled`` is true. The orchestrator's
    validation-time ``cloud_fallback`` remains a separate, older path.
    """

    def __init__(self, primary: MitraAgent, fallback_factory):
        self.primary = primary
        self._fallback_factory = fallback_factory
        self._fallback = None
        self.provider = primary.provider
        self.model_id = primary.model_id
        self.region = primary.region
        self.llm_config = primary.llm_config
        self.last_error = None
        self.used_fallback = False

    def converse(self, message: str) -> str:
        self.used_fallback = False
        try:
            return self.primary.converse(message)
        except ProviderError as e:
            self.last_error = e
            logger.warning("primary provider failed (%s); trying explicit fallback", e.code)
            if self._fallback is None:
                self._fallback = self._fallback_factory()
            self.used_fallback = True
            self.provider = self._fallback.provider
            self.model_id = self._fallback.model_id
            return self._fallback.converse(message)

    def history_mark(self):
        return (self.primary.history_mark(),
                self._fallback.history_mark() if self._fallback is not None else None)

    def rollback(self, mark) -> None:
        primary_mark, fallback_mark = mark if isinstance(mark, tuple) else (mark, None)
        self.primary.rollback(primary_mark)
        if self._fallback is not None:
            self._fallback.rollback(fallback_mark)

    def reset(self) -> None:
        self.primary.reset()
        if self._fallback is not None:
            self._fallback.reset()

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


class MitraAgent:
    def __init__(self, llm_config: dict, tools: list,
                 system_prompt: str = SANSKRIT_SYSTEM_PROMPT, verbose: bool = True):
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
            return str(self._agent(message)).strip()
        except Exception as e:
            err = llm_provider.map_provider_exception(
                e, provider=self.provider, model_id=self.model_id, region=self.region,
            )
            self.last_error = err
            logger.error("LLM provider=%s model=%s failed: %s",
                         self.provider, self.model_id, err)
            raise err from e

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

    def reset(self) -> None:
        self.primary.reset()
        if self._fallback is not None:
            self._fallback.reset()

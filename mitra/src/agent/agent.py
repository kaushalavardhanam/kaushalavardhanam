"""Strands Agent construction (DESIGN §1.3–1.5).

Provider selection is config-driven: local Ollama (Qwen3-VL) by default;
anthropic/bedrock is the Option B one-line swap (FR-6.3). The Strands Agent
owns the conversation loop, message history, and tool dispatch.
"""

from __future__ import annotations

from .prompts import SANSKRIT_SYSTEM_PROMPT

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
        # Strands' default callback handler streams reply tokens straight to
        # stdout — set verbose=False (e.g. in batch/test scripts) to silence
        # it and get only the final string from converse().
        self.max_history_turns = max_history_turns
        agent_kwargs = {} if verbose else {"callback_handler": None}
        self._agent = Agent(
            model=self._make_model(llm_config),
            tools=tools,
            system_prompt=system_prompt,
            **agent_kwargs,
        )

    @staticmethod
    def _make_model(cfg: dict):
        provider = cfg.get("provider", "ollama")
        if provider == "ollama":
            from strands.models.ollama import OllamaModel

            return OllamaModel(
                host=cfg.get("host", "http://localhost:11434"),
                model_id=cfg.get("id", "qwen3-vl:8b"),
                temperature=cfg.get("temperature", 0.3),
                keep_alive=cfg.get("keep_alive", "30m"),
                # Qwen3-VL defaults to thinking mode in Ollama, which burns the
                # latency budget and can leave the reply empty — keep it off.
                additional_args={"think": cfg.get("think", False)},
            )
        if provider == "anthropic":
            from strands.models.anthropic import AnthropicModel

            return AnthropicModel(model_id=cfg["id"])
        if provider == "bedrock":
            from strands.models import BedrockModel

            return BedrockModel(model_id=cfg["id"])
        raise ValueError(f"unknown LLM provider: {provider!r}")

    def converse(self, message: str) -> str:
        """One turn: user message in, final agent text out (tools may run)."""
        reply = str(self._agent(message)).strip()
        self._trim_history()
        return reply

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

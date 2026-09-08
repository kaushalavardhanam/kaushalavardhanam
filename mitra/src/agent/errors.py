"""Actionable provider failures (never silent model/provider swaps)."""

from __future__ import annotations


class ProviderError(Exception):
    """LLM/VLM provider failure with an operator-facing recovery hint."""

    def __init__(self, message: str, *, code: str, provider: str,
                 model_id: str | None = None, region: str | None = None,
                 actionable: str = ""):
        super().__init__(message)
        self.code = code
        self.provider = provider
        self.model_id = model_id
        self.region = region
        self.actionable = actionable

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.actionable:
            parts.append(self.actionable)
        return " ".join(parts)

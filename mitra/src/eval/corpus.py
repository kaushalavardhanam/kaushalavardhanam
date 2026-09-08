"""Load the checked-in evaluation corpora (no private recordings)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2] / "evals" / "corpus"


def load_yaml(name: str) -> dict:
    path = ROOT / name
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def conversation_scenarios() -> list[dict]:
    data = load_yaml("conversation.yaml")
    return list(data["scenarios"])


def recognition_items() -> list[dict]:
    data = load_yaml("recognition.yaml")
    return list(data["items"])


def vision_items() -> list[dict]:
    data = load_yaml("vision.yaml")
    return list(data["items"])

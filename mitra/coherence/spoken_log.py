"""Append-only JSONL logging of every line the agent (or user) speaks."""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "spoken_lines.jsonl"


def _log_path() -> Path:
    override = os.environ.get("MITRA_SPOKEN_LOG")
    return Path(override) if override else DEFAULT_LOG_PATH


def log_spoken_line(text, role="agent", turn_id=None, **extra):
    """Append one spoken line to the JSONL log and return the record."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "turn_id": turn_id,
        "text": text,
    }
    record.update(extra)
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def read_spoken_lines(path=None):
    """Return all logged records as a list of dicts."""
    path = Path(path) if path else _log_path()
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    records = read_spoken_lines()
    if not records:
        print("No spoken lines logged yet at", _log_path())
        sys.exit(0)
    for r in records:
        print(f"[{r['ts']}] {r['role']} (turn {r['turn_id']}): {r['text']}")
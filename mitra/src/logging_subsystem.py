"""Structured per-turn logging (FR-7).

Appends one JSON line per conversation turn (timestamps, language, transcript,
reply, per-stage latency) to ``logs/turns.jsonl``. ``--debug`` mirrors the
conversation on the console, each spoken line followed by its English gloss
(FR-7.2, ``gloss.py``; the gloss is also recorded here as ``reply_en``). No
audio is ever persisted (FR-7.3).
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def setup_logging(debug: bool = False) -> logging.Logger:
    logger = logging.getLogger("mitra")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    for noisy in ("parler_tts", "transformers", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    return logger


class TurnLogger:
    """Accumulates one conversation turn and appends it as a JSON line."""

    def __init__(self, log_dir: str | Path, logger: logging.Logger | None = None):
        self.path = Path(log_dir) / "turns.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger("mitra")
        self._turn: dict = {}

    def start_turn(self) -> None:
        self._turn = {"ts": datetime.now(timezone.utc).isoformat(), "stages": {}}

    def set(self, key: str, value) -> None:
        self._turn[key] = value

    @contextmanager
    def stage(self, name: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self._turn.setdefault("stages", {})[name] = round(time.monotonic() - t0, 3)

    def emit(self) -> dict:
        record = self._turn
        self._turn = {}
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:  # logging must never take the session down (FR-6.4)
            self.logger.exception("failed to write turn log")
        self.logger.debug("turn: %s", json.dumps(record, ensure_ascii=False))
        return record

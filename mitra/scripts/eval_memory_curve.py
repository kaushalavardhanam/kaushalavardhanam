#!/usr/bin/env python3
"""RSS curve over a long inject session (issue #9 §11).

This measures the orchestrator + lexicon + canned agent, not Qwen/Whisper
weights. The 2026-09-16 +950 MB / 10 turns figure is recorded separately
and includes model caches.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def main() -> int:
    try:
        import mitra  # noqa: F401
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "mitra", _ROOT / "src" / "__init__.py",
            submodule_search_locations=[str(_ROOT / "src")],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["mitra"] = module
        spec.loader.exec_module(module)

    from mitra.eval.sanskrit_reference import REFERENCE_REPLIES
    from mitra.lexicon.store import LexiconStore
    from mitra.orchestrator import Event, State
    from mitra.pipeline_trace import memory_mb
    from mitra.robot.reachy import FakeReachy
    from mitra.eval.corpus import conversation_scenarios

    class FakeTTS:
        def synthesize(self, text):
            import numpy as np
            return np.zeros(1600, dtype="float32"), 16000

    class Scripted:
        def __init__(self, replies):
            self.replies = list(replies)

        def converse(self, message):
            if self.replies:
                return self.replies.pop(0)
            return "अहं अत्र अस्मि।"

        def reset(self):
            self.replies = []

    from mitra.orchestrator import Orchestrator

    scenarios = conversation_scenarios()
    replies = [REFERENCE_REPLIES[s["id"]]["sanskrit"] for s in scenarios] * 4
    orch = Orchestrator(
        robot=FakeReachy(), agent=Scripted(replies), tts=FakeTTS(),
        lexicon=LexiconStore(":memory:"),
    )
    orch.state = State.LISTENING
    curve = []
    for i in range(30):
        scenario = scenarios[i % len(scenarios)]
        orch.handle_event(Event("utterance", scenario["expected"]))
        orch.state = State.LISTENING
        curve.append({"turn": i + 1, "rss_mb": memory_mb(), "id": scenario["id"]})
    out = _ROOT / "evals" / "results" / "memory_curve.json"
    out.write_text(json.dumps({
        "n": 30,
        "path": "inject + FakeReachy + canned agent (no model weights)",
        "rss_mb_start": curve[0]["rss_mb"],
        "rss_mb_end": curve[-1]["rss_mb"],
        "delta_mb": (
            None if curve[0]["rss_mb"] is None or curve[-1]["rss_mb"] is None
            else round(curve[-1]["rss_mb"] - curve[0]["rss_mb"], 1)
        ),
        "curve": curve,
        "note": (
            "2026-09-16 live session grew 4120→5070 MB (+950 MB / 10 turns) "
            "including Whisper/TTS caches and excluding the Ollama process. "
            "This inject curve isolates orchestrator/history."
        ),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out} start={curve[0]['rss_mb']} end={curve[-1]['rss_mb']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

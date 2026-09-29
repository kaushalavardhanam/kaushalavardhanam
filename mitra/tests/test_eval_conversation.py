"""Mode A inject helpers used by scripts/eval_conversation.py."""

import importlib.util
from argparse import Namespace
from pathlib import Path

from mitra.eval.sanskrit import aggregate

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_conversation.py"


def _load_eval_conversation():
    spec = importlib.util.spec_from_file_location("eval_conversation_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(orchestrator: str = "custom") -> Namespace:
    return Namespace(
        provider="fixture",
        model_id="cloud-agent-reference",
        region=None,
        orchestrator=orchestrator,
        repeats=1,
    )


def test_inject_custom_and_pipecat_score_reference_replies():
    ev = _load_eval_conversation()
    for engine in ("custom", "pipecat"):
        rows = ev.injected_e2e(_args(engine))
        assert len(rows) == 10
        assert all(r["tts_played"] for r in rows)
        assert all(r["orchestrator"] == engine for r in rows)
        assert all(r["review"]["validator_ok"] for r in rows)
        summary = aggregate([r["review"] for r in rows])
        assert summary["quality_gate"]

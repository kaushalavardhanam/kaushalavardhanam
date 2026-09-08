#!/usr/bin/env python3
"""Conversation evaluation (issue #7 §6).

Mode B (default): send the *expected transcript* to a provider, bypassing
microphone/VAD/ASR. Isolates response quality from recognition.

Mode A: drive the real Orchestrator. ``--inject`` feeds expected transcripts
(post-ASR). ``--spoken`` is the live mic path and needs the Reachy daemon.

    python scripts/eval_conversation.py --mode controlled --provider bedrock \\
        --model-id us.amazon.nova-pro-v1:0
    python scripts/eval_conversation.py --mode end-to-end --inject --robot fake
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))


def _ensure_pkg() -> None:
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


def controlled_run(args) -> list[dict]:
    from mitra.agent.errors import ProviderError
    from mitra.agent.prompts import SANSKRIT_SYSTEM_PROMPT
    from mitra.agent.validator import validate
    from mitra.eval.corpus import conversation_scenarios
    from mitra.eval.cost import estimate_usd
    from mitra.eval.sanskrit import evaluate_response
    from mitra.eval.sanskrit_reference import EVALUATOR_ID

    rows = []
    history: list[dict] = []
    ollama_agent = None
    for scenario in conversation_scenarios():
        prompt = f"[lang={scenario['language']}] {scenario['expected']}"
        t0 = time.monotonic()
        error = None
        error_code = None
        text = ""
        meta = {}
        try:
            if args.provider == "bedrock":
                from mitra.eval.bedrock_converse import converse_text

                meta = converse_text(
                    model_id=args.model_id,
                    user=prompt,
                    system=SANSKRIT_SYSTEM_PROMPT,
                    region=args.region,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    history=history if args.session else None,
                )
                text = meta["text"]
                if args.session:
                    history.append({"role": "user", "content": [{"text": prompt}]})
                    if meta.get("assistant_message"):
                        history.append(meta["assistant_message"])
            elif args.provider == "ollama":
                from mitra.agent.agent import MitraAgent

                if ollama_agent is None or not args.session:
                    ollama_agent = MitraAgent(
                        {"provider": "ollama", "id": args.model_id,
                         "host": args.ollama_host, "temperature": args.temperature},
                        tools=[], verbose=False,
                    )
                text = ollama_agent.converse(prompt)
                meta = {"latency_s": round(time.monotonic() - t0, 3),
                        "model_id": args.model_id, "region": None}
            else:
                raise SystemExit(f"unsupported provider {args.provider}")
        except ProviderError as e:
            error = f"{e.code}: {e}"
            error_code = e.code
            text = ""
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            text = ""
        ok, reason = validate(text) if text else (False, "empty")
        review = evaluate_response(
            prompt=scenario["expected"], sanskrit=text or "",
            evaluator=EVALUATOR_ID,
        )
        row = {
            "prompt": scenario["expected"],
            "id": scenario["id"],
            "run": 1,
            "test_mode": "controlled-session" if args.session else "controlled",
            "asr_transcript": scenario["expected"],
            "provider": args.provider,
            "model": args.model_id,
            "region": args.region,
            "sanskrit": text,
            "validator_ok": ok,
            "validator_reason": reason,
            "latency_s": meta.get("latency_s"),
            "input_tokens": meta.get("input_tokens"),
            "output_tokens": meta.get("output_tokens"),
            "est_usd": estimate_usd(
                args.model_id, meta.get("input_tokens"), meta.get("output_tokens")
            ),
            "error": error,
            "error_code": error_code,
            "review": review,
            "ollama_contacted": args.provider == "ollama",
        }
        rows.append(row)
        print(f"{scenario['id']:10} ok={ok} {text[:60]!r} {error or ''}")
    return rows


def injected_e2e(args) -> list[dict]:
    from mitra.agent import prompts
    from mitra.eval.corpus import conversation_scenarios
    from mitra.lexicon.store import LexiconStore
    from mitra.logging_subsystem import TurnLogger
    from mitra.orchestrator import Event, Orchestrator, State
    from mitra.robot.reachy import FakeReachy

    class ScriptedAgent:
        def __init__(self, replies):
            self.replies = list(replies)
            self.calls = []
            self.provider = args.provider
            self.model_id = args.model_id or "fixture"
            self.region = args.region

        def converse(self, message: str) -> str:
            self.calls.append(message)
            if self.replies:
                return self.replies.pop(0)
            return "अहं अत्र अस्मि।"

        def reset(self):
            pass

    from mitra.eval.sanskrit_reference import REFERENCE_REPLIES

    class FakeTTS:
        def __init__(self):
            self.spoken = []

        def synthesize(self, text):
            import numpy as np
            self.spoken.append(text)
            return np.zeros(1600, dtype="float32"), 16000

    rows = []
    for run in range(1, args.repeats + 1):
        robot = FakeReachy()
        tts = FakeTTS()
        replies = [REFERENCE_REPLIES[s["id"]]["sanskrit"]
                   for s in conversation_scenarios()]
        agent = ScriptedAgent(replies)
        orch = Orchestrator(
            robot=robot, agent=agent, tts=tts, lexicon=LexiconStore(":memory:"),
            llm_meta={"provider": "fixture", "model_id": "cloud-agent-reference"},
        )
        orch.state = State.LISTENING
        for scenario in conversation_scenarios():
            t0 = time.monotonic()
            orch.handle_event(Event("utterance", scenario["expected"]))
            spoken = tts.spoken[-1] if tts.spoken else ""
            rows.append({
                "prompt": scenario["expected"],
                "id": scenario["id"],
                "run": run,
                "test_mode": "end-to-end-inject",
                "asr_transcript": scenario["expected"],
                "provider": "fixture",
                "model": "cloud-agent-reference",
                "sanskrit": spoken,
                "ttfa_s": round(time.monotonic() - t0, 3),
                "tts_played": bool(robot.played),
                "note": "Injected expected transcript through Orchestrator + TTS. "
                        "Not a live microphone run.",
            })
            orch.state = State.LISTENING
    return rows


def write_table(rows: list[dict], path: Path) -> None:
    lines = [
        "| Prompt | Run | Test mode | ASR transcript | Model | Sanskrit | Grammar | Semantic | Gloss matches | Latency | Result |",
        "|---|---:|---|---|---|---|---:|---:|---|---:|---|",
    ]
    for r in rows:
        sans = (r.get("sanskrit") or "").replace("\n", " ")
        review = r.get("review") or {}
        result = r.get("error") or ("pass" if r.get("validator_ok", True) else "fail")
        lines.append(
            f"| {r.get('prompt','')} | {r.get('run',1)} | {r.get('test_mode')} | "
            f"{r.get('asr_transcript','')} | {r.get('model')} | {sans} | "
            f"{review.get('grammar', '')} | {review.get('semantic', '')} | "
            f"{review.get('gloss_agrees', '')} | "
            f"{r.get('latency_s') or r.get('ttfa_s') or ''} | {result} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    _ensure_pkg()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["controlled", "end-to-end"], default="controlled")
    parser.add_argument("--inject", action="store_true",
                        help="Mode A helper: feed expected transcripts (no mic)")
    parser.add_argument("--spoken", action="store_true",
                        help="Mode A live mic (requires daemon + operator)")
    parser.add_argument("--provider", default="bedrock")
    parser.add_argument("--model-id", default="us.amazon.nova-pro-v1:0")
    parser.add_argument("--region", default=None)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--session", action="store_true",
                        help="Mode B: keep conversation history across the ten prompts")
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "evals" / "results" / "conversation.jsonl")
    args = parser.parse_args()

    if args.spoken:
        print("Live spoken Mode A is interactive. Run python main.py --debug "
              "on the Mac with the MuJoCo daemon, say the wake word, then each "
              "corpus question at least three times. This script will not "
              "block on a microphone in unattended CI.")
        return 2

    if args.mode == "controlled":
        rows = controlled_run(args)
    else:
        rows = injected_e2e(args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_table(rows, args.out.with_suffix(".md"))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

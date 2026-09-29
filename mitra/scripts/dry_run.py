#!/usr/bin/env python3
"""Run a conversation through the real pipeline with no robot and no audio.

Everything between the transcript and the spoken text is real — the Ollama
agent, the phrasebook retrieval, the lexicon, the validator, the morphology
checks, the retry — with FakeReachy and a silent TTS standing in for hardware.
That makes the Sanskrit and the checks testable in seconds at a desk, instead
of by talking to a robot for five minutes.

    python3 scripts/dry_run.py                    # the 10-question script
    python3 scripts/dry_run.py --gloss            # + English of every reply
    python3 scripts/dry_run.py -q "Do you play?"  # one question
    python3 scripts/dry_run.py --no-checks        # compare against the old behaviour

    # the 50-question evaluation set, scored (eval/README.md)
    python3 scripts/dry_run.py --questions eval/questions.yaml --log-dir logs/eval
    python3 eval/eval_grammar.py --turns logs/eval/turns.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
try:
    import mitra  # noqa: F401
except ImportError:
    spec = importlib.util.spec_from_file_location(
        "mitra", ROOT / "src" / "__init__.py",
        submodule_search_locations=[str(ROOT / "src")])
    module = importlib.util.module_from_spec(spec)
    sys.modules["mitra"] = module
    spec.loader.exec_module(module)

import yaml  # noqa: E402

# The script from the logged sessions, so runs are comparable turn by turn.
QUESTIONS = [
    "What are you doing?", "Where do you live?", "Do you play?",
    "What's your favorite food?", "What's your favorite subject?",
    "What are you reading?", "Do you listen to music?", "Do you play sports?",
    "What will you do today?", "Will you be my friend?",
]


def load_questions(path: Path) -> list[tuple[str | None, str]]:
    """(question_id, text) pairs from an eval set, or from the default script."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [(q.get("id"), q["text"]) for q in data["questions"]]


class SilentTTS:
    """Records what would have been spoken; synthesizes nothing."""

    def __init__(self):
        self.spoken: list[str] = []

    def synthesize(self, text: str):
        import numpy as np

        self.spoken.append(text)
        return np.zeros(16, dtype=np.float32), 16000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("-q", "--question", action="append", dest="questions")
    ap.add_argument("--gloss", action="store_true",
                    help="translate each reply back to English (a 2nd model call)")
    ap.add_argument("--no-checks", action="store_true",
                    help="disable the morphology checks, to compare")
    ap.add_argument("--questions", metavar="YAML",
                    help="an evaluation set to drive instead of the built-in script")
    ap.add_argument("--log-dir", metavar="DIR",
                    help="write turns.jsonl here, for eval/eval_grammar.py. A "
                         "directory of its own keeps the run separate from the "
                         "live log, which the scorer would otherwise match "
                         "questions against across every past session.")
    args = ap.parse_args()

    from mitra.agent.agent import MitraAgent
    from mitra.agent.tools import build_tools
    from mitra.logging_subsystem import TurnLogger
    from mitra.lexicon.phrasebook import Phrasebook
    from mitra.lexicon.store import LexiconStore
    from mitra.logging_subsystem import setup_logging
    from mitra.orchestrator import Event, Orchestrator, State
    from mitra.robot.reachy import FakeReachy

    main_spec = importlib.util.spec_from_file_location("mitra_main", ROOT / "main.py")
    mitra_main = importlib.util.module_from_spec(main_spec)
    main_spec.loader.exec_module(mitra_main)

    config = yaml.safe_load(open(args.config, encoding="utf-8"))
    logger = setup_logging(True)
    if args.no_checks:
        config.setdefault("sanskrit", {})["enabled"] = False

    class EvalTurnLogger(TurnLogger):
        """TurnLogger that stamps each record with the question it answers.

        The scorer otherwise has to guess which turn belongs to which question
        by fuzzy-matching the transcript, which is exactly the guesswork the
        eval set's README asks to remove.
        """

        question_id: str | None = None

        def start_turn(self) -> None:
            super().start_turn()
            if self.question_id:
                self.set("question_id", self.question_id)

    robot, tts = FakeReachy(), SilentTTS()
    turn_logger = EvalTurnLogger(args.log_dir, logger) if args.log_dir else None
    phrasebook = Phrasebook(
        config.get("phrasebook", {}).get("path", str(ROOT / "data/phrasebook.jsonl")))
    checker = mitra_main._build_grammar_checker(config, phrasebook, logger)

    glosser = None
    if args.gloss:
        from mitra.gloss import GLOSS_SYSTEM_PROMPT, Glosser

        glosser = Glosser(
            lambda: MitraAgent(config["models"]["llm"], [],
                               system_prompt=GLOSS_SYSTEM_PROMPT,
                               verbose=False, max_history_turns=0), logger=logger)

    orchestrator = Orchestrator(
        robot=robot, agent=MitraAgent(config["models"]["llm"],
                                      build_tools(robot, tts), verbose=False),
        tts=tts, lexicon=LexiconStore(config["lexicon"]["db_path"]),
        phrasebook=phrasebook, glosser=glosser, grammar_checker=checker,
        turn_logger=turn_logger, logger=logger, gestures=False,
        max_reply_chars=config["session"]["max_reply_chars"],
        max_sentences=config["session"].get("max_sentences", 1),
    )

    if args.questions:
        questions = load_questions(Path(args.questions))
    else:
        questions = [(None, q) for q in (args.question or QUESTIONS)]

    print(f"\n{'='*72}\nchecks: "
          f"{', '.join(checker.checks) if checker else 'DISABLED'}\n{'='*72}")
    started = time.monotonic()
    for question_id, question in questions:
        if turn_logger is not None:
            turn_logger.question_id = question_id
        orchestrator.state = State.LISTENING
        spoken_before = len(tts.spoken)
        turn = time.monotonic()
        orchestrator.handle_event(Event("utterance", question))
        reply = tts.spoken[-1] if len(tts.spoken) > spoken_before else "(nothing)"
        label = f"{question_id}  " if question_id else ""
        print(f"\n  {label}you:   {question}")
        print(f"  {' ' * len(label)}mitra: {reply}   [{time.monotonic() - turn:.1f}s]")
    print(f"\n{len(questions)} turns in {time.monotonic() - started:.0f}s")
    if turn_logger is not None:
        print(f"turn log: {turn_logger.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

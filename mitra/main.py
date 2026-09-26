#!/usr/bin/env python3
"""Mitra entry point: component wiring + run loop (DESIGN §2).

    python main.py --check          # report which components are available
    python main.py                  # run against the reachy daemon (real or --sim)
    python main.py --robot fake     # run without any robot daemon
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Quiet two harmless-but-scary warnings from the ML stack: the tokenizers
# fork warning, and transformers' full-config dumps during Parler-TTS load.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # no "Fetching 4 files" spam

_ROOT = Path(__file__).resolve().parent


def _ensure_package() -> None:
    """Make ``import mitra`` resolve to ./src when not pip-installed."""
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


_ensure_package()

import yaml  # noqa: E402

from mitra.logging_subsystem import TurnLogger, setup_logging  # noqa: E402


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def check(config: dict) -> int:
    """Report availability of every optional layer (Phase 0 helper).

    Speech probes follow the configured backend rather than a fixed list: the
    mlx/parler stack is Apple-Silicon-only, so on Linux those are *expected*
    to be absent and reporting them as MISSING sends people chasing a
    non-problem. Only the layer this config will actually load is checked.
    """
    import importlib
    import json
    import urllib.request

    def probe(name: str, module: str) -> bool:
        try:
            importlib.import_module(module)
            print(f"  ok       {name}")
            return True
        except ImportError as e:
            print(f"  MISSING  {name}  ({e.name or e})")
            return False

    models = config["models"]
    asr_backend = models["asr"].get("backend", "mlx")
    wake_backend = models["wake"].get("backend", "mlx")
    tts_engine = models["tts"].get("engine", "indic-parler-tts")

    print("components:")
    probe("reachy-mini (robot/sim)", "reachy_mini")
    probe("strands-agents (agent)", "strands")
    probe("openwakeword (wake)", "openwakeword")
    probe("silero-vad (VAD)", "silero_vad")

    if asr_backend == "mlx" or wake_backend == "mlx":
        probe(f"mlx-whisper (ASR, backend={asr_backend})", "mlx_whisper")
    else:
        probe(f"transformers (ASR, backend={asr_backend})", "transformers")

    if tts_engine == "indic-parler-tts":
        probe(f"parler-tts (TTS, engine={tts_engine})", "parler_tts")
    else:
        probe(f"transformers (TTS, engine={tts_engine})", "transformers")

    host = models["llm"]["host"]
    model_id = models["llm"]["id"]
    print("ollama:")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as resp:
            installed = [m["name"] for m in json.load(resp).get("models", [])]
        state = "ok" if any(m.startswith(model_id) for m in installed) else "MISSING"
        print(f"  ok       server at {host}")
        print(f"  {state:8} model {model_id}  (installed: {', '.join(installed) or 'none'})")
    except OSError as e:
        print(f"  DOWN     {host}  ({e})")

    from mitra.lexicon.phrasebook import Phrasebook
    from mitra.lexicon.store import LexiconStore

    store = LexiconStore()  # in-memory, seeds from the bundled JSON
    print(f"lexicon: {store.count()} seed entries")

    from mitra.lexicon.dictionary import Dictionary
    from mitra.sanskrit import Analyzer

    settings = config.get("sanskrit", {})
    analyzer = Analyzer(settings.get("data_dir", "data/vidyut"))
    if analyzer.available:
        from mitra.lexicon.vocabulary import Vocabulary

        vocabulary = Vocabulary(analyzer, seed_path=_ROOT / "src" / "lexicon" /
                                "seed_lexicon.json")
        print(f"sanskrit: kosha ok, {len(vocabulary.lemmas)} allowed lemmas, "
              f"checks {', '.join(settings.get('checks', ()))}")
    else:
        print("sanskrit: MISSING morphology data — replies are checked for "
              "script only.\n  pip install 'mitra[sanskrit]' && "
              "python3 scripts/fetch_sanskrit_data.py")
    dictionary = Dictionary()
    print(f"dictionary: {'ok' if dictionary.available else 'MISSING'} "
          f"(Cologne MW/Apte, used by mitra-lexicon)")

    pb_path = config.get("phrasebook", {}).get("path", "data/phrasebook.jsonl")
    phrasebook = Phrasebook(pb_path)
    if phrasebook.count():
        print(f"phrasebook: {phrasebook.count()} entries ({pb_path})")
    else:
        print(f"phrasebook: MISSING at {pb_path} — replies will be ungrounded.\n"
              f"  build it: python3 scripts/build_phrasebook.py data/daily.pdf")
    return 0


def _build_grammar_checker(config: dict, phrasebook, logger):
    """Analyzer + vocabulary + enabled checks, or None if switched off.

    Every failure here is non-fatal by design: the checks are an addition to
    the deterministic validator, not a prerequisite for speaking (FR-6.4).
    """
    settings = config.get("sanskrit", {})
    if not settings.get("enabled", True):
        return None
    from mitra.lexicon.vocabulary import Vocabulary
    from mitra.sanskrit import Analyzer
    from mitra.sanskrit.grammar import DEFAULT_CHECKS, Checker

    analyzer = Analyzer(settings.get("data_dir", "data/vidyut"), logger)
    if not analyzer.available:
        return None
    extra: tuple[str, ...] = ()
    if settings.get("ground_in_phrasebook", True) and phrasebook is not None:
        extra = tuple(phrasebook.sentences())
    vocabulary = Vocabulary(
        analyzer, seed_path=Path(__file__).resolve().parent / "src" /
        "lexicon" / "seed_lexicon.json",
        extra_texts=extra, logger_=logger)
    checks = tuple(settings.get("checks", DEFAULT_CHECKS))
    logger.info("sanskrit checks enabled: %s", ", ".join(checks) or "none")
    return Checker(analyzer, vocabulary, checks)


def build_and_run(config: dict, robot_backend: str, debug: bool) -> int:
    debug = debug or config["logging"].get("debug", False)
    logger = setup_logging(debug)

    from mitra.agent.agent import MitraAgent
    from mitra.agent.tools import build_tools
    from mitra.audio.asr import Transcriber
    from mitra.audio.vad import make_segmenter
    from mitra.audio.wake import make_wake_detector
    from mitra.lexicon.phrasebook import Phrasebook
    from mitra.lexicon.store import LexiconStore
    from mitra.orchestrator import Orchestrator
    from mitra.speech.tts import SanskritTTS

    models = config["models"]
    tts_kwargs = {}
    if models["tts"].get("voice_description"):
        tts_kwargs["voice_description"] = models["tts"]["voice_description"]
    tts = SanskritTTS(model=models["tts"]["model"], device=models["tts"]["device"],
                      engine=models["tts"].get("engine", "indic-parler-tts"),
                      fallback_model=models["tts"].get("fallback", "facebook/mms-tts-hin"),
                      **tts_kwargs)
    # Warm up TTS at startup for the same reason as ASR below: the Parler
    # voice is a ~3.8 GB one-time download and a slow first load — without
    # this, the robot goes silent exactly when it should first greet. The VITS
    # path (engine: vits) is far smaller, but the warmup still hides its load.
    logger.info("warming up TTS (first run downloads the voice)...")
    try:
        tts.synthesize("नमस्ते")
    except Exception:
        logger.exception("TTS warmup failed — continuing; speech will retry")
    wake = make_wake_detector(**models["wake"])
    if hasattr(wake, "warmup"):
        logger.info("warming up wake ASR (first run downloads whisper-small)...")
        wake.warmup()
    vad_cfg = models["vad"]
    segmenter = make_segmenter(
        vad_cfg.get("engine", "silero"),
        min_silence_s=vad_cfg.get("min_silence_s", 0.8),
        max_utterance_s=vad_cfg.get("max_utterance_s", 15.0),
    )
    asr = Transcriber(default_model=models["asr"]["default"],
                      sanskrit_model=models["asr"].get("sanskrit"),
                      backend=models["asr"].get("backend", "mlx"),
                      device=models["asr"].get("device", "mps"))
    # Warm up ASR before the run loop: Whisper large-v3 (~3 GB) downloads on
    # first use. Without this, the download would stall the FIRST conversation
    # turn for minutes with no feedback; here it happens at startup with a log
    # line, and later runs load from the local cache in seconds.
    logger.info("warming up ASR (first run downloads Whisper, ~1.6 GB one time)...")
    import numpy as np
    try:
        asr.transcribe(np.zeros(8000, dtype=np.float32))  # 0.5 s of silence
    except Exception:
        logger.exception("ASR warmup failed — continuing; the first turn will retry")
    lexicon = LexiconStore(config["lexicon"]["db_path"])

    # Retrieval corpus for conversational grounding. Loaded before the robot
    # connects so a missing or empty file is visible in the log ahead of the
    # first turn rather than being silently absent from every reply.
    phrasebook = Phrasebook(
        config.get("phrasebook", {}).get("path", "data/phrasebook.jsonl"))
    if not phrasebook.count():
        logger.warning("phrasebook empty — replies will be ungrounded. Build it "
                       "with: python3 scripts/build_phrasebook.py data/daily.pdf")

    # Connect to the robot ONLY after all model warmups: opening the daemon
    # connection starts the microphone pipeline, and the multi-GB model loads
    # above starve the audio threads badly enough that GStreamer floods the
    # console with "Can't record audio fast enough" and drops samples. With
    # the mic opened last, warmups happen in silence and listening starts
    # with everything already resident.
    if robot_backend == "fake":
        from mitra.robot.reachy import FakeReachy

        robot = FakeReachy()
        logger.warning("using FakeReachy — no camera/audio/motion")
    else:
        from mitra.robot.reachy import ReachyRobot

        mic_source = config["robot"].get("mic_source", "robot")
        if mic_source == "built_in":
            logger.warning("mic_source=built_in: listening through the host's "
                           "own mic, not the robot's — camera/speaker/motion "
                           "still go through it (macOS: reachy_mini#820; "
                           "Linux: missing GStreamer webrtc plugin)")
        logger.info("connecting to the robot daemon...")
        robot = ReachyRobot(
            mic_chunk_s=config["robot"].get("mic_chunk_s", 0.08),
            mic_source=mic_source,
            built_in_mic_device=config["robot"].get(
                "built_in_mic_device", "MacBook Pro Microphone"),
        )

    # verbose=False silences Strands' default callback handler, which streams
    # reply tokens straight to stdout and interleaves them with the log lines
    # ("क2026-08-08 23:07:56 INFO mitra: speak: ..."). The --debug log already
    # carries every reply, so nothing is lost.
    agent = MitraAgent(
        models["llm"], build_tools(robot, tts), verbose=False,
        max_history_turns=config.get("agent", {}).get("max_history_turns", 4))

    fallback_factory = None
    cloud = config.get("cloud_fallback", {})
    if cloud.get("enabled") and cloud.get("provider"):  # FR-6.3
        fallback_factory = lambda: MitraAgent(  # noqa: E731
            {"provider": cloud["provider"], "id": cloud["model_id"]},
            build_tools(robot, tts), verbose=False,
            max_history_turns=config.get("agent", {}).get("max_history_turns", 4),
        )

    # Morphology checks over every reply (DESIGN §5). Built after the
    # phrasebook so the vocabulary can absorb it, and before the agent so a
    # missing dataset is reported once at startup rather than per turn.
    grammar_checker = _build_grammar_checker(config, phrasebook, logger)

    # Debug runs mirror every spoken line in English (FR-7.2). It is a second,
    # history-free call to the same Ollama model — no extra VRAM, and it runs
    # after playback starts, so it never delays speech. Off outside --debug,
    # and switchable there with logging.gloss_english.
    glosser = None
    if debug and config["logging"].get("gloss_english", True):
        from mitra.gloss import GLOSS_SYSTEM_PROMPT, Glosser

        glosser = Glosser(
            lambda: MitraAgent(models["llm"], [],
                               system_prompt=GLOSS_SYSTEM_PROMPT,
                               verbose=False, max_history_turns=0),
            logger=logger,
        )

    orchestrator = Orchestrator(
        robot=robot, agent=agent, tts=tts, lexicon=lexicon,
        wake=wake, segmenter=segmenter, asr=asr, phrasebook=phrasebook,
        turn_logger=TurnLogger(config["logging"]["dir"], logger),
        glosser=glosser,
        grammar_checker=grammar_checker,
        logger=logger,
        gestures=config["robot"].get("gestures", True),
        silence_timeout_s=config["session"]["silence_timeout_s"],
        max_reply_chars=config["session"]["max_reply_chars"],
        max_sentences=config["session"].get("max_sentences", 1),
        fallback_agent_factory=fallback_factory,
    )
    try:
        orchestrator.run()
    except KeyboardInterrupt:
        pass
    finally:
        orchestrator.stop()
        robot.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="mitra")
    parser.add_argument("--config", default=str(_ROOT / "config.yaml"))
    parser.add_argument("--debug", action="store_true",
                        help="mirror the conversation on the console (FR-7.2)")
    parser.add_argument("--check", action="store_true",
                        help="report component availability and exit")
    parser.add_argument("--robot", choices=["reachy", "fake"], default=None,
                        help="override robot backend from config")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.check:
        return check(config)
    backend = args.robot or config["robot"].get("backend", "reachy")
    return build_and_run(config, backend, args.debug)


if __name__ == "__main__":
    raise SystemExit(main())

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
    """Report availability of every optional layer (Phase 0 helper)."""
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

    print("components:")
    probe("reachy-mini (robot/sim)", "reachy_mini")
    probe("strands-agents (agent)", "strands")
    probe("openwakeword (wake)", "openwakeword")
    probe("silero-vad (VAD)", "silero_vad")
    probe("mlx-whisper (ASR)", "mlx_whisper")
    probe("parler-tts (TTS)", "parler_tts")
    probe("pipecat (optional orchestrator)", "pipecat")

    llm = config["models"]["llm"]
    from mitra.agent import provider as llm_provider

    identity = llm_provider.describe_llm(llm)
    print(f"llm:      provider={identity['provider']} model={identity['model_id']}")
    if identity["provider"] == "bedrock":
        print("ollama:   SKIPPED  (bedrock mode — Ollama/Qwen is not contacted)")
        status = llm_provider.bedrock_credential_status()
        region = identity.get("region") or "(unset — set AWS_REGION or models.llm.region)"
        print(f"bedrock:  region={region} credentials={'ok' if status['ok'] else 'MISSING'}")
        if not status["ok"]:
            print(f"           {status.get('reason')} {status.get('hint', '')}")
        else:
            print(f"           method={status.get('method')} key={status.get('access_key_fp')}")
    else:
        host = llm.get("host", "http://localhost:11434")
        model_id = llm.get("id")
        print("ollama:")
        try:
            with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as resp:
                models = [m["name"] for m in json.load(resp).get("models", [])]
            state = "ok" if any(m.startswith(model_id) for m in models) else "MISSING"
            print(f"  ok       server at {host}")
            print(f"  {state:8} model {model_id}  (installed: {', '.join(models) or 'none'})")
        except OSError as e:
            print(f"  DOWN     {host}  ({e})")

    from mitra.lexicon.store import LexiconStore

    store = LexiconStore()  # in-memory, seeds from the bundled JSON
    print(f"lexicon: {store.count()} seed entries")
    return 0


def build_and_run(config: dict, robot_backend: str, debug: bool) -> int:
    logger = setup_logging(debug or config["logging"].get("debug", False))

    from mitra.agent.agent import ExplicitFallbackAgent, MitraAgent
    from mitra.agent import provider as llm_provider
    from mitra.agent.tools import build_tools
    from mitra.audio.asr import Transcriber
    from mitra.audio.vad import make_segmenter
    from mitra.audio.wake import make_wake_detector
    from mitra.lexicon.store import LexiconStore
    from mitra.speech.tts import SanskritTTS

    models = config["models"]
    tts_kwargs = {}
    if models["tts"].get("voice_description"):
        tts_kwargs["voice_description"] = models["tts"]["voice_description"]
    tts = SanskritTTS(model=models["tts"]["model"], device=models["tts"]["device"],
                      fallback_model=models["tts"].get("fallback", "facebook/mms-tts-hin"),
                      **tts_kwargs)
    # Warm up TTS at startup for the same reason as ASR below: the Parler
    # voice is a ~3.8 GB one-time download and a slow first load — without
    # this, the robot goes silent exactly when it should first greet.
    logger.info("warming up TTS (first run downloads the voice, ~3.8 GB one time)...")
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
        min_silence_s=vad_cfg.get("min_silence_s", 0.7),
        max_utterance_s=vad_cfg.get("max_utterance_s", 15.0),
        min_speech_s=vad_cfg.get("min_speech_s", 0.25),
        preroll_s=vad_cfg.get("preroll_s", 0.25),
    )
    asr_cfg = models["asr"]
    asr = Transcriber(
        default_model=asr_cfg["default"],
        sanskrit_model=asr_cfg.get("sanskrit"),
        backend=asr_cfg.get("backend", "mlx"),
        device=asr_cfg.get("device", "mps"),
        initial_prompt=asr_cfg.get("initial_prompt"),
        condition_on_previous_text=asr_cfg.get("condition_on_previous_text", False),
        no_speech_threshold=asr_cfg.get("no_speech_threshold", 0.6),
        compression_ratio_threshold=asr_cfg.get("compression_ratio_threshold", 2.4),
        min_peak=asr_cfg.get("min_peak", 0.008),
        filter_hallucinations=asr_cfg.get("filter_hallucinations", True),
        english_retry=asr_cfg.get("english_retry", True),
    )
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
            logger.warning("mic_source=built_in: listening through the Mac's "
                           "own mic (workaround for reachy_mini#820), not the "
                           "robot's — camera/speaker/motion still go through it")
        logger.info("connecting to the robot daemon...")
        robot = ReachyRobot(
            mic_chunk_s=config["robot"].get("mic_chunk_s", 0.08),
            mic_source=mic_source,
            built_in_mic_device=config["robot"].get(
                "built_in_mic_device", "MacBook Pro Microphone"),
        )

    llm_cfg = models["llm"]
    identity = llm_provider.describe_llm(llm_cfg)
    logger.info("LLM %s", identity)
    if identity["provider"] == "bedrock":
        logger.info("bedrock mode active — Ollama/Qwen will not be started or contacted")

    tools = build_tools(robot, tts)
    agent = MitraAgent(llm_cfg, tools)
    explicit_fb = llm_provider.fallback_config(llm_cfg)
    if explicit_fb:
        logger.warning("explicit LLM fallback enabled: %s/%s (not silent)",
                       explicit_fb["provider"], explicit_fb["id"])
        agent = ExplicitFallbackAgent(
            agent, lambda: MitraAgent(explicit_fb, tools),  # noqa: E731
        )

    fallback_factory = None
    cloud = config.get("cloud_fallback", {})
    if cloud.get("enabled") and cloud.get("provider"):  # FR-6.3
        fallback_factory = lambda: MitraAgent(  # noqa: E731
            {"provider": cloud["provider"], "id": cloud["model_id"]},
            tools,
        )

    engine = config.get("orchestration", {}).get("engine", "custom")
    if engine == "pipecat":
        from mitra.orchestration.pipecat_runtime import PipecatOrchestrator

        OrchestratorCls = PipecatOrchestrator
        logger.info("orchestration engine=pipecat (proof of concept)")
    elif engine == "custom":
        from mitra.orchestrator import Orchestrator

        OrchestratorCls = Orchestrator
        logger.info("orchestration engine=custom")
    else:
        raise ValueError(f"unknown orchestration.engine: {engine!r} (custom|pipecat)")

    orchestrator = OrchestratorCls(
        robot=robot, agent=agent, tts=tts, lexicon=lexicon,
        wake=wake, segmenter=segmenter, asr=asr,
        turn_logger=TurnLogger(config["logging"]["dir"], logger),
        logger=logger,
        gestures=config["robot"].get("gestures", True),
        silence_timeout_s=config["session"]["silence_timeout_s"],
        max_reply_chars=config["session"]["max_reply_chars"],
        fallback_agent_factory=fallback_factory,
        llm_meta=identity,
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
    parser.add_argument("--orchestrator", choices=["custom", "pipecat"], default=None,
                        help="override orchestration.engine")
    parser.add_argument("--llm-provider", choices=["ollama", "bedrock", "anthropic"],
                        default=None, help="override models.llm.provider")
    parser.add_argument("--llm-id", default=None, help="override models.llm.id")
    parser.add_argument("--llm-region", default=None, help="override models.llm.region")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.orchestrator:
        config.setdefault("orchestration", {})["engine"] = args.orchestrator
    if args.llm_provider:
        config["models"]["llm"]["provider"] = args.llm_provider
    if args.llm_id:
        config["models"]["llm"]["id"] = args.llm_id
    if args.llm_region:
        config["models"]["llm"]["region"] = args.llm_region
    if args.check:
        return check(config)
    backend = args.robot or config["robot"].get("backend", "reachy")
    return build_and_run(config, backend, args.debug)


if __name__ == "__main__":
    raise SystemExit(main())

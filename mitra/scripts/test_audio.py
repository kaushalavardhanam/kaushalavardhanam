#!/usr/bin/env python3
"""Record and replay voice clips against Mitra's wake/ASR pipeline offline.

Use this when live wake detection seems unreliable ("hey mitra" doesn't
trigger): capture one clip once, then repeatedly test it against the
detector's actual logic without talking into the mic each time. This
isolates whether the problem is mic capture, mic volume, or the wake match
itself — and shows exactly what Whisper heard.

    # Record 3s from the robot's real mic (daemon must be running)
    python scripts/test_audio.py --record 3 --out clips/hello.wav

    # Analyze any WAV/audio file — any samplerate/channels — against the
    # wake detector and the main ASR, printing full diagnostics. Works with
    # a recording made anywhere (phone voice memo, etc.), not just --record.
    python scripts/test_audio.py --file clips/hello.wav

    # Record then immediately analyze
    python scripts/test_audio.py --record 3 --out clips/hello.wav --analyze
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[1]
try:
    import mitra  # noqa: F401
except ImportError:
    spec = importlib.util.spec_from_file_location(
        "mitra", _ROOT / "src" / "__init__.py",
        submodule_search_locations=[str(_ROOT / "src")])
    module = importlib.util.module_from_spec(spec)
    sys.modules["mitra"] = module
    spec.loader.exec_module(module)

from mitra.audio import TARGET_SAMPLERATE, resample  # noqa: E402
from mitra.audio.vad import EnergySegmenter  # noqa: E402
from mitra.audio.wake import TranscriptWakeDetector  # noqa: E402


def record(seconds: float, out_path: pathlib.Path) -> None:
    """Capture using the same mic_source as config.yaml (issue #7)."""
    import soundfile as sf

    from mitra.robot.reachy import ReachyRobot

    cfg = _load_app_config()
    robot_cfg = cfg.get("robot") or {}
    mic_source = robot_cfg.get("mic_source", "robot")
    device = robot_cfg.get("built_in_mic_device", "MacBook Pro Microphone")
    print(f"connecting to the robot daemon (mic_source={mic_source})...")
    robot = ReachyRobot(mic_source=mic_source, built_in_mic_device=device)
    try:
        print(f"recording {seconds:.1f}s in 2s... speak after the beep-less pause")
        time.sleep(2.0)
        print("* recording now *")
        chunks, total = [], 0
        target = int(robot.mic_samplerate * seconds)
        t0 = time.monotonic()
        while total < target and time.monotonic() - t0 < seconds + 5:
            chunk = robot.mic_read()
            if len(chunk):
                chunks.append(chunk)
                total += len(chunk)
        print("* done *")
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, robot.mic_samplerate)
        print(f"saved {len(audio) / robot.mic_samplerate:.1f}s to {out_path} "
              f"@ {robot.mic_samplerate} Hz")
    finally:
        robot.close()


def load_16k_mono(path: pathlib.Path) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != TARGET_SAMPLERATE:
        audio = resample(audio, sr, TARGET_SAMPLERATE)
    return audio.astype(np.float32)


def _load_app_config() -> dict:
    import yaml

    cfg_path = _ROOT / "config.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def analyze(path: pathlib.Path) -> None:
    audio = load_16k_mono(path)
    seconds = len(audio) / TARGET_SAMPLERATE
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    print(f"\n=== {path} ===")
    print(f"duration: {seconds:.2f}s   rms: {rms:.4f}   peak: {peak:.4f}")

    gate = max(EnergySegmenter()._min_gate, rms)  # rough reference, not live-adaptive
    print(f"(EnergySegmenter min_gate={EnergySegmenter()._min_gate:.4f} — "
          f"{'would likely open the speech gate' if rms >= EnergySegmenter()._min_gate else 'BELOW the gate — raise mic volume or move closer'})")

    config = _load_app_config()
    wake_cfg = (config.get("models") or {}).get("wake") or {}
    asr_cfg = (config.get("models") or {}).get("asr") or {}
    wake = TranscriptWakeDetector(
        phrase=wake_cfg.get("phrase", "mitra"),
        asr_model=wake_cfg.get("asr_model", "mlx-community/whisper-small-mlx"),
    )
    print(f"\n--- wake detector ({wake._asr_model.rsplit('/', 1)[-1]}) ---")
    text = wake._mlx_transcribe(audio)
    cleaned = __import__("re").sub(r"[^\wऀ-ॿ]+", "", text.lower())
    matched = [v for v in wake._variants if v in cleaned]
    print(f"heard: {text.strip()!r}")
    print(f"match: {'WAKE (' + ', '.join(matched) + ')' if matched else 'no match'}")

    asr_name = asr_cfg.get("default", "mlx-community/whisper-large-v3-turbo")
    print(f"\n--- main ASR ({asr_name.rsplit('/', 1)[-1]}) [same settings as config.yaml] ---")
    from mitra import language_detector
    from mitra.audio.asr import Transcriber

    asr = Transcriber(
        default_model=asr_name,
        sanskrit_model=asr_cfg.get("sanskrit"),
        backend=asr_cfg.get("backend", "mlx"),
        device=asr_cfg.get("device", "mps"),
        initial_prompt=asr_cfg.get("initial_prompt"),
        condition_on_previous_text=asr_cfg.get("condition_on_previous_text", False),
        min_peak=asr_cfg.get("min_peak", 0.008),
        filter_hallucinations=asr_cfg.get("filter_hallucinations", True),
        english_retry=asr_cfg.get("english_retry", True),
        cpu_model=asr_cfg.get("cpu_model"),
    )
    asr_text, hint = asr.transcribe(audio)
    lang = language_detector.detect(asr_text, hint)
    print(f"transcript: {asr_text.strip()!r}")
    print(f"language: {lang} (asr hint: {hint})")
    print(f"diag: {asr.last_diag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=float, metavar="SECONDS",
                        help="record this many seconds from the robot's mic")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("clips/clip.wav"),
                        help="where to save the recording (default: clips/clip.wav)")
    parser.add_argument("--file", type=pathlib.Path,
                        help="analyze an existing audio file instead of recording")
    parser.add_argument("--analyze", action="store_true",
                        help="also analyze immediately after --record")
    args = parser.parse_args()

    if args.record is not None:
        record(args.record, args.out)
        if args.analyze:
            analyze(args.out)
    elif args.file is not None:
        analyze(args.file)
    else:
        parser.error("pass --record SECONDS or --file PATH")


if __name__ == "__main__":
    main()

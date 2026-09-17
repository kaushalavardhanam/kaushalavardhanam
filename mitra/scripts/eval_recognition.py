#!/usr/bin/env python3
"""Measure ASR accuracy against the recognition corpus (issue #7 §2).

    python scripts/eval_recognition.py                  # print corpus + WER helper
    python scripts/eval_recognition.py --audio-dir clips

WAV files are optional, consented, and never required in git. Names:
``{id}.wav`` or ``{id}-1.wav``. Uses the same Transcriber settings as config.yaml.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import yaml  # noqa: E402


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


def load_wav_16k(path: Path):
    import numpy as np
    import soundfile as sf

    from mitra.audio import TARGET_SAMPLERATE, resample

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != TARGET_SAMPLERATE:
        audio = resample(audio, sr, TARGET_SAMPLERATE)
    return np.asarray(audio, dtype=np.float32)


def transcriber_from_config(config: dict):
    from mitra.audio.asr import Transcriber

    asr = config["models"]["asr"]
    return Transcriber(
        default_model=asr["default"],
        sanskrit_model=asr.get("sanskrit"),
        backend=asr.get("backend", "mlx"),
        device=asr.get("device", "mps"),
        initial_prompt=asr.get("initial_prompt"),
        condition_on_previous_text=asr.get("condition_on_previous_text", False),
        no_speech_threshold=asr.get("no_speech_threshold", 0.6),
        compression_ratio_threshold=asr.get("compression_ratio_threshold", 2.4),
        min_peak=asr.get("min_peak", 0.008),
        filter_hallucinations=asr.get("filter_hallucinations", True),
        english_retry=asr.get("english_retry", True),
        cpu_model=asr.get("cpu_model"),
    )


def main() -> int:
    _ensure_pkg()
    from mitra.eval.corpus import recognition_items
    from mitra.eval.metrics import cer, meaning_preserved, wer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(_ROOT / "config.yaml"))
    parser.add_argument("--audio-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "evals" / "results" / "recognition.jsonl")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    items = recognition_items()
    print(f"{len(items)} corpus items. audio_dir={args.audio_dir}")

    asr = None
    rows = []
    if args.audio_dir:
        asr = transcriber_from_config(config)
        from mitra import language_detector

    for item in items:
        expected = item["expected"]
        matches = []
        if args.audio_dir and args.audio_dir.exists():
            matches = sorted(args.audio_dir.glob(f"{item['id']}*.wav"))
        if not matches:
            rows.append({
                "id": item["id"], "expected": expected, "hypothesis": None,
                "status": "no_audio", "note": "consented WAV not supplied",
            })
            continue
        for wav in matches:
            text, hint = asr.transcribe(load_wav_16k(wav))
            lang = language_detector.detect(text, hint)
            row = {
                "id": item["id"],
                "file": str(wav),
                "expected": expected,
                "hypothesis": text,
                "lang": lang,
                "asr_hint": hint,
                "wer": round(wer(expected, text), 3),
                "cer": round(cer(expected, text), 3),
                "meaning_preserved": meaning_preserved(expected, text),
                "diag": getattr(asr, "last_diag", {}),
            }
            rows.append(row)
            print(f"{item['id']:10} WER={row['wer']:.2f}  {text!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    scored = [r for r in rows if r.get("hypothesis") is not None]
    if scored:
        mean_wer = sum(r["wer"] for r in scored) / len(scored)
        print(f"mean WER {mean_wer:.3f} over {len(scored)} clips")
    else:
        print("No clips scored. Record consented WAVs and re-run with --audio-dir.")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

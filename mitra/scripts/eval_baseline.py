#!/usr/bin/env python3
"""Record a pre-change baseline snapshot (issue #7 §1).

Does not start Ollama or Bedrock. Does not persist microphone audio.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

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

import yaml  # noqa: E402

from mitra.pipeline_trace import memory_mb  # noqa: E402


def git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *cmd], cwd=_ROOT.parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def main() -> int:
    config = yaml.safe_load((_ROOT / "config.yaml").read_text(encoding="utf-8"))
    snapshot = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "git_commit": git(["rev-parse", "HEAD"]),
        "git_describe": git(["describe", "--always", "--dirty"]),
        "branch": git(["branch", "--show-current"]),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "rss_mb": memory_mb(),
        "robot": {
            "backend": config["robot"].get("backend"),
            "mic_source": config["robot"].get("mic_source"),
            "built_in_mic_device": config["robot"].get("built_in_mic_device"),
            "mic_chunk_s": config["robot"].get("mic_chunk_s"),
        },
        "audio": {
            "target_samplerate": 16000,
            "asr": config["models"]["asr"],
            "vad": config["models"]["vad"],
            "wake": {k: v for k, v in config["models"]["wake"].items()
                     if k != "model" or True},
        },
        "llm": {
            "provider": config["models"]["llm"].get("provider"),
            "id": config["models"]["llm"].get("id"),
            "region": config["models"]["llm"].get("region"),
            "fallback_enabled": bool(
                (config["models"]["llm"].get("fallback") or {}).get("enabled")
            ),
        },
        "orchestration": config.get("orchestration", {"engine": "custom"}),
        "env_region": os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"),
        "note": "Raw microphone audio is not recorded. Live latency/memory of a "
                "spoken turn requires the Mac + daemon path.",
    }
    out_dir = _ROOT / "evals" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "baseline.json"
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    print(json.dumps(snapshot, indent=2, ensure_ascii=False))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

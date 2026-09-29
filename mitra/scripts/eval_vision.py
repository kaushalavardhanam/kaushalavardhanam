#!/usr/bin/env python3
"""Vision/tool comparison on identical images (issue #7 §7).

    python scripts/eval_vision.py --image-dir evals/fixtures/vision \\
        --provider bedrock --model-id us.amazon.nova-pro-v1:0

Images are not committed. Capture them from the MuJoCo minimal scene
(apple / croissant / duck) and pass the directory of {id}.jpg files.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def main() -> int:
    _ensure_pkg()
    from mitra.agent.prompts import SANSKRIT_SYSTEM_PROMPT, VISION_JSON_INSTRUCTION
    from mitra.eval.corpus import vision_items
    from mitra.lexicon.store import LexiconStore
    from mitra.orchestrator import _extract_json, Orchestrator

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--provider", default="bedrock")
    parser.add_argument("--model-id", default="us.amazon.nova-pro-v1:0")
    parser.add_argument("--region", default=None)
    parser.add_argument("--out", type=Path,
                        default=_ROOT / "evals" / "results" / "vision.jsonl")
    args = parser.parse_args()

    lexicon = LexiconStore(":memory:")
    rows = []
    for item in vision_items():
        jpeg = None
        if args.image_dir:
            for ext in (".jpg", ".jpeg", ".png"):
                p = args.image_dir / f"{item['id']}{ext}"
                if p.exists():
                    jpeg = p.read_bytes()
                    break
        if jpeg is None:
            rows.append({**item, "status": "no_image",
                         "note": "identical image not supplied"})
            continue
        if args.provider != "bedrock":
            rows.append({**item, "status": "skipped",
                         "note": "scripted Bedrock path only in this helper"})
            continue
        from mitra.eval.bedrock_converse import converse_text

        user = f"[lang=en] {VISION_JSON_INSTRUCTION}\nWhat is this?"
        try:
            meta = converse_text(
                model_id=args.model_id, user=user,
                system=SANSKRIT_SYSTEM_PROMPT, region=args.region,
                images=[jpeg],
            )
            raw = meta["text"]
            err = None
        except Exception as e:
            raw, meta, err = "", {}, f"{type(e).__name__}: {e}"
        # Reuse lexicon substitution without a live orchestrator instance
        class _Lex:
            lexicon = lexicon
            def _apply_lexicon(self, reply):
                return Orchestrator._apply_lexicon(self, reply)
        spoken = _Lex()._apply_lexicon(raw) if raw else ""
        parsed = _extract_json(raw) if raw else None
        rows.append({
            **item,
            "model": args.model_id,
            "raw": raw,
            "spoken": spoken,
            "parsed": parsed,
            "lexicon_overrode": bool(
                item.get("verified_sa") and item["verified_sa"] in spoken
            ),
            "latency_s": meta.get("latency_s"),
            "error": err,
        })
        print(f"{item['id']:12} {spoken!r} {err or ''}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

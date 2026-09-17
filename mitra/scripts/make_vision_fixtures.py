#!/usr/bin/env python3
"""Generate synthetic stand-in JPEGs for the vision bake-off.

These are NOT MuJoCo captures. They are identical across candidates so
object-ID / lexicon / grounding can be compared. Label them as synthetic
in every results table. Do not treat a pass here as a simulator pass.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "evals" / "fixtures" / "vision"


def _apple(size=256) -> Image.Image:
    im = Image.new("RGB", (size, size), (236, 240, 230))
    d = ImageDraw.Draw(im)
    d.ellipse((58, 62, 198, 210), fill=(196, 42, 42), outline=(120, 20, 20), width=3)
    d.rectangle((122, 40, 134, 72), fill=(90, 60, 30))
    d.ellipse((128, 36, 168, 64), fill=(46, 120, 52))
    return im


def _croissant(size=256) -> Image.Image:
    im = Image.new("RGB", (size, size), (245, 236, 220))
    d = ImageDraw.Draw(im)
    d.pieslice((30, 70, 226, 230), 200, 340, fill=(214, 160, 72), outline=(150, 100, 40))
    d.arc((60, 95, 196, 205), 210, 330, fill=(180, 120, 50), width=6)
    return im


def _duck(size=256) -> Image.Image:
    im = Image.new("RGB", (size, size), (210, 228, 240))
    d = ImageDraw.Draw(im)
    d.ellipse((60, 90, 190, 190), fill=(240, 205, 50), outline=(170, 140, 20), width=3)
    d.ellipse((150, 60, 214, 124), fill=(240, 205, 50), outline=(170, 140, 20), width=3)
    d.polygon([(210, 88), (246, 100), (210, 112)], fill=(230, 120, 30))
    d.ellipse((188, 80, 200, 92), fill=(30, 30, 30))
    return im


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("apple", _apple), ("croissant", _croissant), ("duck", _duck)):
        path = OUT / f"{name}.jpg"
        fn().save(path, "JPEG", quality=90)
        print(f"wrote {path}")
    (OUT / "README.md").write_text(
        "Synthetic stand-in JPEGs for identical-image VLM comparison.\n"
        "Not captured from the MuJoCo `minimal` scene. Do not commit private photos.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Simple helpers for arranging images on a grid."""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


def _load_image(path: Path, tile_size: Tuple[int, int]) -> Optional[Image.Image]:
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(tile_size, Image.LANCZOS)
        return img
    except Exception:
        return None


def make_image_grid(
    image_paths: Sequence[str],
    rows: int = 4,
    cols: int = 4,
    tile_size: Tuple[int, int] = (256, 256),
    captions: Optional[Sequence[str]] = None,
    background_color: Tuple[int, int, int] = (30, 30, 30),
) -> Image.Image:
    """Return a PIL image that lays out the provided images on a grid."""
    if not image_paths:
        raise ValueError("No image paths provided for grid creation.")

    tile_width, tile_height = tile_size
    canvas = Image.new(
        "RGB",
        (cols * tile_width, rows * tile_height),
        color=background_color,
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    idx = 0
    for row in range(rows):
        for col in range(cols):
            if idx >= len(image_paths):
                break
            img = _load_image(Path(image_paths[idx]), tile_size)
            if img is not None:
                canvas.paste(img, (col * tile_width, row * tile_height))
                if captions:
                    caption = captions[idx]
                    text_pos = (col * tile_width + 4, row * tile_height + tile_height - 12)
                    draw.text(text_pos, caption, fill=(255, 255, 255), font=font)
            idx += 1
        if idx >= len(image_paths):
            break

    return canvas

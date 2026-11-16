"""Image operations for the four experiments (all deterministic, generated on the fly)."""
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from common import IMAGES, SEED
from habitats import HABITAT_COLORS, HABITATS

LEVELS = {"full": 1.0, "loose": 0.5, "medium": 0.25, "tight": 0.05, "ultra_tight": 0.0}
MISMATCH_BLUR, MISMATCH_NOISE_STD, BG_BLUR = 12, 10, 30


def load_image(file_name: str) -> Image.Image:
    return Image.open(IMAGES / file_name).convert("RGB")


def scaled_box(img: Image.Image, x, y, w, h, json_w, json_h) -> tuple[int, int, int, int]:
    """JSON bbox [x,y,w,h] (pixels of the JSON-size image) -> integer (l,t,r,b) in the ACTUAL image."""
    sx, sy = img.width / json_w, img.height / json_h
    l, t = math.floor(x * sx), math.floor(y * sy)
    r, b = math.ceil((x + w) * sx), math.ceil((y + h) * sy)
    l, t = max(0, l), max(0, t)
    r, b = min(img.width, max(r, l + 1)), min(img.height, max(b, t + 1))
    return l, t, r, b


def margin_crop(img: Image.Image, box, m: float) -> Image.Image:
    l, t, r, b = box
    w, h = r - l, b - t
    L, T = max(0, math.floor(l - m * w)), max(0, math.floor(t - m * h))
    R, B = min(img.width, math.ceil(r + m * w)), min(img.height, math.ceil(b + m * h))
    return img.crop((L, T, R, B))


@lru_cache(maxsize=32)
def _blurred_solid(habitat: str, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.new("RGB", size, HABITAT_COLORS[habitat]).filter(
        ImageFilter.GaussianBlur(MISMATCH_BLUR)), dtype=np.float32)


def synthetic_composite(img: Image.Image, box, habitat: str, sample_id: int) -> Image.Image:
    """Full-size solid colour -> Gaussian blur r=12 -> Gaussian noise std=10; animal pasted at the SAME position."""
    base = _blurred_solid(habitat, img.size)
    rng = np.random.default_rng([SEED, int(sample_id), HABITATS.index(habitat)])
    bg = Image.fromarray(np.clip(base + rng.normal(0.0, MISMATCH_NOISE_STD, base.shape), 0, 255).astype(np.uint8))
    bg.paste(img.crop(box), (box[0], box[1]))
    return bg


def background_only(img: Image.Image, box) -> Image.Image:
    out = img.copy()
    out.paste(img.crop(box).filter(ImageFilter.GaussianBlur(BG_BLUR)), (box[0], box[1]))
    return out


def make_variant(img: Image.Image, box, variant: str, sample_id: int) -> Image.Image:
    if variant == "orig":
        return img
    if variant.startswith("crop_"):
        return margin_crop(img, box, LEVELS[variant[5:]])
    if variant.startswith("mismatch_"):
        return synthetic_composite(img, box, variant[9:], sample_id)
    if variant == "bgonly":
        return background_only(img, box)
    raise ValueError(variant)

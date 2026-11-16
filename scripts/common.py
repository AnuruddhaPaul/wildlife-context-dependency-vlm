"""Shared constants/paths for the context-dependency reproduction."""
import os
import random
from pathlib import Path

import numpy as np

# Model weights already cached on the HDD (same HF repos open_clip would download); used read-only.
HF_LOCAL_CACHE = Path("D:/HUG_MODELs")
if HF_LOCAL_CACHE.exists() and "HF_HUB_CACHE" not in os.environ:
    os.environ["HF_HUB_CACHE"] = str(HF_LOCAL_CACHE)
    os.environ["HF_HUB_OFFLINE"] = "1"

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
IMAGES = DATA / "images"
CACHE = DATA / "cache"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"

SPECIES = ["badger", "bird", "bobcat", "car", "cat", "coyote", "deer", "dog", "fox", "lizard",
           "mountain_lion", "opossum", "rabbit", "raccoon", "rodent", "skunk", "squirrel"]
EXPECTED_COUNTS = {s: 100 for s in SPECIES} | {"badger": 28, "mountain_lion": 44}
MAX_PER_SPECIES = 100


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def class_name(species: str) -> str:
    return species.replace("_", " ")

"""Fetch the full-resolution CCT images needed for the sample (resumable).

Source order per file: (1) local data/images, (2) D:/cct_images (full-res copy already on the HDD),
(3) LILA Azure blob (caltech-unzipped/cct_images, URL pattern verified with HEAD requests).
Every file is opened with PIL and its size compared with width/height from the JSON.
"""
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

from common import DATA, IMAGES, LOGS

HDD_FULL = Path("D:/cct_images")
BASE = "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images"


def ok(path: Path, w: int, h: int) -> bool:
    try:
        with Image.open(path) as im:
            im.load()
            return im.size == (w, h)
    except Exception:
        return False


def fetch(row) -> tuple[str, str]:
    dst = IMAGES / row.file_name
    if dst.exists() and ok(dst, row.json_w, row.json_h):
        return row.file_name, "cached"
    src = HDD_FULL / row.file_name
    if src.exists():
        shutil.copyfile(src, dst)
        if ok(dst, row.json_w, row.json_h):
            return row.file_name, "copied_from_hdd"
        dst.unlink()
    for attempt in range(4):
        try:
            tmp = dst.with_suffix(".part")
            with urllib.request.urlopen(f"{BASE}/{row.file_name}", timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(dst)
            return row.file_name, "downloaded" if ok(dst, row.json_w, row.json_h) else "downloaded_SIZE_MISMATCH"
        except Exception:
            time.sleep(2 * (attempt + 1))
    return row.file_name, "FAILED"


if __name__ == "__main__":
    IMAGES.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    s = pd.read_csv(DATA / "sample.csv").drop_duplicates("file_name")
    rows = list(s.itertuples())
    t0, status = time.time(), {}
    with ThreadPoolExecutor(8) as ex:
        for i, (name, st) in enumerate(ex.map(fetch, rows), 1):
            status[name] = st
            if i % 100 == 0 or i == len(rows):
                print(f"{i}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
    summary = pd.Series(status).value_counts()
    print(summary.to_string())
    pd.Series(status).to_csv(LOGS / "fetch_status.csv")
    bad = [n for n, st in status.items() if st in ("FAILED", "downloaded_SIZE_MISMATCH")]
    print("problems:", bad or "none")

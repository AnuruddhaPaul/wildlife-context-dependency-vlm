"""Download CCT metadata + eccv_18_all_images_sm from LILA BC (resumable, stdlib only).

URLs were taken from https://lila.science/datasets/caltech-camera-traps/ and each
one was verified with an HTTP HEAD request (status 200) before use.
NOTE: 'caltech_images_20210113.json' is NOT published on that page and returns 404;
the bbox file LILA actually lists is caltech_bboxes_20200316.json.
"""
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
BASE = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps"
FILES = {
    "caltech_bboxes_20200316.json": f"{BASE}/labels/caltech_bboxes_20200316.json",
    "caltech_camera_traps.json.zip": f"{BASE}/labels/caltech_camera_traps.json.zip",
    "eccv_18_all_images_sm.tar.gz": f"{BASE}/eccv_18_all_images_sm.tar.gz",
}


def download(url: str, dst: Path, chunk: int = 1 << 20) -> None:
    head = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(head, timeout=60) as r:
        total = int(r.headers["Content-Length"])
    have = dst.stat().st_size if dst.exists() else 0
    if have == total:
        print(f"[skip] {dst.name} already complete ({total / 1e6:.1f} MB)")
        return
    if have > total:
        dst.unlink()
        have = 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    t0, last = time.time(), 0.0
    with urllib.request.urlopen(req, timeout=60) as r, open(dst, "ab" if have else "wb") as f:
        got = have
        while buf := r.read(chunk):
            f.write(buf)
            got += len(buf)
            if time.time() - last > 15:
                rate = (got - have) / max(time.time() - t0, 1e-9) / 1e6
                print(f"  {dst.name}: {got / 1e6:,.0f}/{total / 1e6:,.0f} MB ({rate:.1f} MB/s)", flush=True)
                last = time.time()
    assert dst.stat().st_size == total, f"size mismatch for {dst.name}"
    print(f"[done] {dst.name} ({total / 1e6:.1f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:] or list(FILES)
    for name in only:
        download(FILES[name], RAW / name)

"""Check how much of the CCT bbox metadata is covered by images already on the HDD."""
import json
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "data" / "raw" / "caltech_bboxes_20200316.json").read_text())
meta = {i["file_name"]: i for i in d["images"]}
ann_imgs = {a["image_id"] for a in d["annotations"]}
need = {i["file_name"] for i in d["images"] if i["id"] in ann_imgs}

for folder in (Path("D:/cct_images"), Path("D:/data/cct20")):
    print(f"\n=== {folder}")
    if not folder.exists():
        print("missing")
        continue
    files = [p for p in folder.rglob("*") if p.is_file()]
    exts = Counter(p.suffix.lower() for p in files)
    print("files:", len(files), "| extensions:", dict(exts.most_common(6)))
    print("subfolders (first 10):", [p.name for p in folder.iterdir() if p.is_dir()][:10])
    jpgs = {p.name: p for p in files if p.suffix.lower() in (".jpg", ".jpeg")}
    hit = [n for n in jpgs if n in meta]
    print("jpgs:", len(jpgs), "| matching bbox-JSON file_names:", len(hit), "| of images-with-boxes covered:", len(set(hit) & need), "/", len(need))
    sizes = Counter()
    ratios = Counter()
    for n in hit[:200]:
        with Image.open(jpgs[n]) as im:
            sizes[im.size] += 1
            m = meta[n]
            ratios[round(im.size[0] / m["width"], 3)] += 1
    print("actual sizes (first 200 matched):", sizes.most_common(6))
    print("actual_w / json_w ratios:", ratios.most_common(6))

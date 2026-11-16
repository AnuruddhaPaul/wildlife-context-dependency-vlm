"""Diagnose why badger / mountain_lion image counts differ from the expected 28 / 44."""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "data" / "raw" / "caltech_bboxes_20200316.json").read_text())
cats = {c["id"]: c["name"] for c in d["categories"]}
imgs = {i["id"]: i for i in d["images"]}
by_img = defaultdict(list)
for a in d["annotations"]:
    by_img[a["image_id"]].append(cats[a["category_id"]])

for sp in ("badger", "mountain_lion"):
    n_ann = sum(1 for labs in by_img.values() for l in labs if l == sp)
    img_any = [i for i, labs in by_img.items() if sp in labs]
    img_first = [i for i, labs in by_img.items() if labs[0] == sp]
    img_only = [i for i, labs in by_img.items() if labs == [sp]]
    img_all_same = [i for i, labs in by_img.items() if set(labs) == {sp}]
    mixed = Counter(tuple(sorted(by_img[i])) for i in img_any if set(by_img[i]) != {sp})
    print(f"{sp}: annotations={n_ann} | images with >=1 such box={len(img_any)} | first box is {sp}={len(img_first)} "
          f"| exactly one box={len(img_only)} | all boxes {sp}={len(img_all_same)}")
    print(f"   multi-label images containing {sp}: {dict(mixed)}")
    print(f"   box counts per such image: {Counter(len(by_img[i]) for i in img_any)}")
    print(f"   image sizes: {Counter((imgs[i]['width'], imgs[i]['height']) for i in img_any)}")

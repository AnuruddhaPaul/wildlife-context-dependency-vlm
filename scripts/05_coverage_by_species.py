"""Per-species box coverage of the HDD image folders (eccv_18 sm set vs full-res cct_images)."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECIES = ["badger", "bird", "bobcat", "car", "cat", "coyote", "deer", "dog", "fox", "lizard",
           "mountain_lion", "opossum", "rabbit", "raccoon", "rodent", "skunk", "squirrel"]
d = json.loads((ROOT / "data" / "raw" / "caltech_bboxes_20200316.json").read_text())
cats = {c["id"]: c["name"] for c in d["categories"]}
fname = {i["id"]: i["file_name"] for i in d["images"]}

sm = {p.name for p in Path("D:/data/cct20/images").glob("*.jpg")}
full = {p.name for p in Path("D:/cct_images").glob("*.jpg")}

tot, in_sm, in_full, in_either = Counter(), Counter(), Counter(), Counter()
for a in d["annotations"]:
    s = cats[a["category_id"]]
    if s not in SPECIES:
        continue
    f = fname[a["image_id"]]
    tot[s] += 1
    in_sm[s] += f in sm
    in_full[s] += f in full
    in_either[s] += (f in sm) or (f in full)

print(f"{'species':14s} {'boxes':>7s} {'in eccv_sm':>11s} {'in cct_images':>14s} {'either':>7s}")
for s in SPECIES:
    print(f"{s:14s} {tot[s]:7d} {in_sm[s]:11d} {in_full[s]:14d} {in_either[s]:7d}")

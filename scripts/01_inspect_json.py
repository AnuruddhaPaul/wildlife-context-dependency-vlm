"""Inspect the CCT bbox metadata: keys, categories, bbox format, per-image box counts."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
d = json.loads((ROOT / "data" / "raw" / "caltech_bboxes_20200316.json").read_text())
print("top-level keys:", list(d))
print("n images:", len(d["images"]), "| n annotations:", len(d["annotations"]), "| n categories:", len(d["categories"]))
print("image[0]:", d["images"][0])
print("annotation[0]:", d["annotations"][0])
cats = {c["id"]: c["name"] for c in d["categories"]}
print("categories:", cats)

per_img = Counter(a["image_id"] for a in d["annotations"])
print("images with >=1 annotation:", len(per_img), "| box-count histogram:", sorted(Counter(per_img.values()).items()))
has_bbox = Counter(("bbox" in a) for a in d["annotations"])
print("annotations with bbox key:", dict(has_bbox))
print("annotations per category:", {cats[k]: v for k, v in sorted(Counter(a["category_id"] for a in d["annotations"]).items())})
print("unique image sizes (w,h):", Counter((i.get("width"), i.get("height")) for i in d["images"]).most_common(8))

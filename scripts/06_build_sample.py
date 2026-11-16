"""Build the 1,572-sample evaluation set: up to 100 boxes per species, seed 42 (per-BOX sampling).

Each sample is one bounding-box annotation. badger (28) and mountain_lion (44) have fewer than 100
boxes, so all of them are used; a few images therefore appear more than once with different boxes.
"""
import json

import numpy as np
import pandas as pd

from common import EXPECTED_COUNTS, MAX_PER_SPECIES, RAW, DATA, SEED, SPECIES

d = json.loads((RAW / "caltech_bboxes_20200316.json").read_text())
cats = {c["id"]: c["name"] for c in d["categories"]}
imgs = {i["id"]: i for i in d["images"]}

rows = []
for a in d["annotations"]:
    sp = cats[a["category_id"]]
    if sp in SPECIES:
        im = imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        rows.append(dict(species=sp, ann_id=a["id"], image_id=a["image_id"], file_name=im["file_name"],
                         x=x, y=y, w=w, h=h, json_w=im["width"], json_h=im["height"]))
pool = pd.DataFrame(rows).sort_values(["species", "ann_id"]).reset_index(drop=True)
n_boxes_in_img = pool.groupby("image_id")["ann_id"].transform("count")
all_boxes_per_img = pd.Series([0] * len(pool))
pool["n_species_boxes_in_image"] = n_boxes_in_img

rng = np.random.default_rng(SEED)
parts = []
for sp in SPECIES:  # alphabetical, fixed order -> deterministic
    sub = pool[pool.species == sp]
    k = min(MAX_PER_SPECIES, len(sub))
    parts.append(sub.iloc[np.sort(rng.choice(len(sub), size=k, replace=False))])
sample = pd.concat(parts).reset_index(drop=True)
sample.insert(0, "sample_id", range(len(sample)))

counts = sample.species.value_counts().to_dict()
print("per-species counts:", {s: counts[s] for s in SPECIES})
print("TOTAL:", len(sample), "| expected:", sum(EXPECTED_COUNTS.values()))
bad = {s: (counts[s], EXPECTED_COUNTS[s]) for s in SPECIES if counts[s] != EXPECTED_COUNTS[s]}
print("MISMATCH vs expected:", bad or "none")
print("unique images:", sample.file_name.nunique(), "| images used by >1 sample:", int((sample.file_name.value_counts() > 1).sum()))
sample.to_csv(DATA / "sample.csv", index=False)
print("wrote", DATA / "sample.csv")

"""Count candidate images per species under different filtering rules (before sampling)."""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECIES = ["badger", "bird", "bobcat", "car", "cat", "coyote", "deer", "dog", "fox", "lizard",
           "mountain_lion", "opossum", "rabbit", "raccoon", "rodent", "skunk", "squirrel"]
EXPECTED = {s: 100 for s in SPECIES} | {"badger": 28, "mountain_lion": 44}

d = json.loads((ROOT / "data" / "raw" / "caltech_bboxes_20200316.json").read_text())
cats = {c["id"]: c["name"] for c in d["categories"]}
by_img = defaultdict(list)
for a in d["annotations"]:
    by_img[a["image_id"]].append(cats[a["category_id"]])

rule_a, rule_b = Counter(), Counter()  # a: exactly one box in image; b: >=1 box, all one species-agnostic, first box is species
for img, labs in by_img.items():
    if len(labs) == 1 and labs[0] in SPECIES:
        rule_a[labs[0]] += 1
    if labs[0] in SPECIES:
        rule_b[labs[0]] += 1

print(f"{'species':14s} {'exactly-1-box':>14s} {'first-box>=1':>14s}  -> min(100, .)  expected")
tot_a = tot_b = 0
for s in SPECIES:
    a, b = min(100, rule_a[s]), min(100, rule_b[s])
    tot_a, tot_b = tot_a + a, tot_b + b
    print(f"{s:14s} {rule_a[s]:14d} {rule_b[s]:14d}  -> {a:4d} / {b:4d}    {EXPECTED[s]}")
print(f"TOTAL sampled: rule A (exactly one box) = {tot_a} | rule B (first box) = {tot_b} | expected = {sum(EXPECTED.values())}")

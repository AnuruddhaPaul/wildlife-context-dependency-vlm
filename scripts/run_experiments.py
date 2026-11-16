"""Run Experiments 1-4 (zero-shot, no training). Usage: python run_experiments.py [--smoke]

Every image variant (5 crops, original, 4 synthetic-habitat composites, background-only) is generated once and
encoded by all three models in a single pass; L2-normalised image embeddings are cached per (model, variant) so the
run is resumable. Predictions for each prompt are then derived from the cached embeddings.
"""
import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import CACHE, DATA, EXPECTED_COUNTS, IMAGES, RESULTS, SEED, SPECIES, set_seed
from habitats import EXPECTED_HABITAT_TOTALS, HABITATS, check_totals, habitats_for
from imaging import LEVELS, load_image, make_variant, scaled_box
from models import MODEL_SPECS, PROMPTS, ZeroShotModel

BATCH = 64
SMOKE_SPECIES = ["dog", "opossum", "raccoon"]


class VariantDS(Dataset):
    def __init__(self, rows, variant, preprocs):
        self.rows, self.variant, self.preprocs = rows, variant, preprocs

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = load_image(r.file_name)
        box = scaled_box(img, r.x, r.y, r.w, r.h, r.json_w, r.json_h)
        pil = make_variant(img, box, self.variant, r.sample_id)
        return tuple(p(pil) for p in self.preprocs)


def slug(label: str) -> str:
    return label.replace("/", "").replace("-", "")


def pick_bg_only(sample: pd.DataFrame, per_species: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    parts = []
    for sp in SPECIES:
        sub = sample[sample.species == sp]
        if len(sub):
            parts.append(sub.iloc[np.sort(rng.choice(len(sub), size=min(per_species, len(sub)), replace=False))])
    return pd.concat(parts)


def variant_table(sample: pd.DataFrame, bg: pd.DataFrame) -> dict[str, pd.DataFrame]:
    v = {f"crop_{lvl}": sample for lvl in LEVELS}
    v["orig"] = sample
    for h in HABITATS:
        v[f"mismatch_{h}"] = sample[sample.species.map(lambda s: h in habitats_for(s))]
    v["bgonly"] = bg
    return {k: df for k, df in v.items() if len(df)}  # e.g. no urban species in the 3-species smoke subset


def main(smoke: bool, plain_gelu: bool = False):
    # default run = QuickGELU forced (results/); --plain-gelu = literal open_clip 4.x call (results/plain_gelu/)
    out_dir = RESULTS / "smoke" if smoke else (RESULTS / "plain_gelu" if plain_gelu else RESULTS)
    cache = CACHE / ("smoke" if smoke else ("plain_gelu" if plain_gelu else "full"))
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    set_seed()
    timings, t_all = {}, time.time()

    env = json.loads((RESULTS / "env_info.json").read_text())
    device = env["device_used"]
    print(f"device: {device} ({env.get('gpu_name', 'cpu')}) | smoke={smoke}")

    sample = pd.read_csv(DATA / "sample.csv")
    if smoke:
        sample = pd.concat([sample[sample.species == s].head(10) for s in SMOKE_SPECIES]).reset_index(drop=True)
    else:
        counts = sample.species.value_counts().to_dict()
        assert counts == {s: EXPECTED_COUNTS[s] for s in counts} and len(sample) == 1572, "sample counts deviate"
        assert check_totals(counts) == EXPECTED_HABITAT_TOTALS, check_totals(counts)
    missing = [f for f in sample.file_name.unique() if not (IMAGES / f).exists()]
    assert not missing, f"{len(missing)} images missing, run 07_fetch_images.py first"
    bg = pick_bg_only(sample)
    variants = variant_table(sample, bg)
    n_mm = sum(len(df) for k, df in variants.items() if k.startswith("mismatch_"))
    print(f"samples={len(sample)} | mismatch predictions per model={n_mm} | bg-only n={len(bg)}")
    if not smoke:
        assert n_mm == 3144

    # ---- load models ----
    t0 = time.time()
    models = {lab: ZeroShotModel(lab, device, force_quick_gelu=not plain_gelu) for lab in MODEL_SPECS}
    timings["load_models_s"] = round(time.time() - t0, 1)
    print("models loaded in", timings["load_models_s"], "s")

    # ---- image embeddings ----
    workers = 0 if smoke else max(2, min(6, (os.cpu_count() or 4) - 2))
    for vname, df in variants.items():
        need = [lab for lab in models if not (cache / f"{slug(lab)}__{vname}.npz").exists()]
        if not need:
            print(f"[skip] {vname} cached")
            continue
        t0 = time.time()
        rows = [SimpleNamespace(**r) for r in df.to_dict("records")]  # picklable for Windows DataLoader workers
        ds = VariantDS(rows, vname, [models[l].preprocess for l in need])
        dl = DataLoader(ds, batch_size=BATCH, num_workers=workers, pin_memory=(device == "cuda"))
        feats = {l: [] for l in need}
        for batch in tqdm(dl, desc=vname, leave=False):
            for lab, x in zip(need, batch):
                feats[lab].append(models[lab].encode_images(x))
        for lab in need:
            np.savez(cache / f"{slug(lab)}__{vname}.npz", sample_ids=df.sample_id.to_numpy(), feats=np.concatenate(feats[lab]))
        timings[f"embed_{vname}_s"] = round(time.time() - t0, 1)
        print(f"[done] {vname}: n={len(df)} in {timings[f'embed_{vname}_s']} s", flush=True)

    # ---- predictions from cached embeddings ----
    t0 = time.time()
    spec = sample.set_index("sample_id")
    plan = [("exp1", f"crop_{lvl}", lvl, "neutral") for lvl in LEVELS]
    plan += [("exp2_baseline", "orig", "original", "neutral")]
    plan += [("exp3", "orig", "original", p) for p in PROMPTS]
    plan += [("exp2_mismatch", f"mismatch_{h}", h, "neutral") for h in HABITATS]
    plan += [("exp4", "bgonly", "bg_only_blur30", "neutral")]
    plan = [t for t in plan if t[1] in variants]
    recs = []
    for lab, m in models.items():
        for exp, vname, cond, prompt in plan:
            z = np.load(cache / f"{slug(lab)}__{vname}.npz")
            pred, conf = m.classify(z["feats"], prompt)
            ids = z["sample_ids"]
            truth = spec.loc[ids, "species"].to_numpy()
            pred_sp = np.array(SPECIES)[pred]
            recs.append(pd.DataFrame({
                "model": lab, "experiment": exp, "condition": cond, "prompt": prompt,
                "habitat": cond if exp == "exp2_mismatch" else "", "sample_id": ids, "species": truth,
                "image_path": [str(IMAGES / f) for f in spec.loc[ids, "file_name"]],
                "prediction": pred_sp, "confidence": conf, "correct": pred_sp == truth}))
    allp = pd.concat(recs, ignore_index=True)
    allp.to_csv(out_dir / "all_predictions.csv", index=False)
    timings["predictions_s"] = round(time.time() - t0, 1)
    timings["total_s"] = round(time.time() - t_all, 1)
    meta = {"logit_scale": {l: m.logit_scale for l, m in models.items()}, "model_info": {l: m.info for l, m in models.items()},
            "hf_hub_cache": os.environ.get("HF_HUB_CACHE", "default"), "timings": timings,
            "batch_size": BATCH, "n_samples": len(sample), "smoke": smoke}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(allp)} prediction rows -> {out_dir / 'all_predictions.csv'} | total {timings['total_s']} s")
    print(allp[allp.experiment.isin(['exp2_baseline', 'exp1'])].groupby(["model", "experiment", "condition"]).correct.mean().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plain-gelu", action="store_true")
    a = ap.parse_args()
    main(a.smoke, a.plain_gelu)

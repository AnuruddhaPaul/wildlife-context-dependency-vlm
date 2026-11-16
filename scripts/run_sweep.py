"""Animal-only (16 classes, no car) experiments x activation-function sweep. Usage: python run_sweep.py [--smoke]

All five activations are swapped (no retraining) into every MLP block (vision + text tower) of the three models through
one shared SwitchAct module per model, so each image variant is generated ONCE and encoded for all 15 (model, activation)
pairs. Embeddings are cached per (model, activation, variant) -> resumable.
"""
import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from open_clip.transformer import QuickGELU
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import CACHE, DATA, IMAGES, RESULTS, SPECIES, class_name, set_seed
from habitats import HABITATS, check_totals
from imaging import LEVELS
from models import MODEL_SPECS, PROMPTS
from run_experiments import BATCH, SMOKE_SPECIES, VariantDS, pick_bg_only, slug, variant_table

CLASSES = [s for s in SPECIES if s != "car"]  # 16 animal classes
ACTS = ["quickgelu", "gelu", "gelu_tanh", "silu", "relu"]
NATIVE = {"ViT-B/32": "quickgelu", "ViT-L/14": "quickgelu", "BioCLIP": "gelu"}  # activation each model was trained with
EXPECTED_ANIMAL_HABITATS = {"arctic": 1172, "desert": 572, "aquatic": 900, "urban": 300}


class SwitchAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode = "quickgelu"

    def forward(self, x):
        m = self.mode
        if m == "quickgelu":
            return x * torch.sigmoid(1.702 * x)
        if m == "gelu":
            return F.gelu(x)
        if m == "gelu_tanh":
            return F.gelu(x, approximate="tanh")
        if m == "silu":
            return F.silu(x)
        if m == "relu":
            return F.relu(x)
        raise ValueError(m)


def load_model(label: str, device: str):
    spec = MODEL_SPECS[label]
    kw = {"pretrained": spec["pretrained"]} if spec["pretrained"] else {}
    if spec["quick_gelu"]:
        kw["force_quick_gelu"] = True  # only to avoid the warning; the activation is replaced below anyway
    model, _, pre = open_clip.create_model_and_transforms(spec["name"], **kw)
    tok = open_clip.get_tokenizer(spec["name"])
    model.to(device).eval()
    act, n = SwitchAct(), 0
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, (nn.GELU, QuickGELU)):
                setattr(mod, name, act)
                n += 1
    assert n > 0, "no activation layers found to patch"
    return dict(model=model, pre=pre, tok=tok, act=act, n_patched=n, scale=float(model.logit_scale.exp().item()))


@torch.no_grad()
def text_emb(m, mode, prompt, device, _cache={}):
    key = (id(m["model"]), mode, prompt)
    if key not in _cache:
        m["act"].mode = mode
        f = m["model"].encode_text(m["tok"]([PROMPTS[prompt].format(class_name(s)) for s in CLASSES]).to(device))
        _cache[key] = f / f.norm(dim=-1, keepdim=True)
    return _cache[key]


def main(smoke: bool):
    tag = "animal_only_smoke" if smoke else "animal_only"
    out_dir, cache = RESULTS / tag, CACHE / ("sweep_smoke" if smoke else "sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    set_seed()
    timings, t_all = {}, time.time()
    env = json.loads((RESULTS / "env_info.json").read_text())
    device = env["device_used"]

    full17 = pd.read_csv(DATA / "sample.csv")
    bg_all = pick_bg_only(full17)  # same 85 images as the 17-class run ...
    sample = full17[full17.species != "car"].reset_index(drop=True)
    bg = bg_all[bg_all.species != "car"]  # ... minus the 5 car images
    if smoke:
        sample = pd.concat([sample[sample.species == s].head(10) for s in SMOKE_SPECIES]).reset_index(drop=True)
        bg = pick_bg_only(sample)
    else:
        counts = sample.species.value_counts().to_dict()
        assert len(sample) == 1472 and "car" not in counts, len(sample)
        assert check_totals(counts) == EXPECTED_ANIMAL_HABITATS, check_totals(counts)
        assert len(bg) == 80
    missing = [f for f in sample.file_name.unique() if not (IMAGES / f).exists()]
    assert not missing, f"{len(missing)} images missing"
    variants = variant_table(sample, bg)
    n_mm = sum(len(d) for k, d in variants.items() if k.startswith("mismatch_"))
    print(f"device={device} smoke={smoke} | samples={len(sample)} | mismatch preds/model/activation={n_mm} | bg-only n={len(bg)}", flush=True)
    if not smoke:
        assert n_mm == 2944

    t0 = time.time()
    M = {lab: load_model(lab, device) for lab in MODEL_SPECS}
    timings["load_models_s"] = round(time.time() - t0, 1)
    print("patched activation layers:", {l: m["n_patched"] for l, m in M.items()}, flush=True)

    workers = 0 if smoke else max(2, min(6, (os.cpu_count() or 4) - 2))
    for vname, df in variants.items():
        need = [(l, a) for l in M for a in ACTS if not (cache / f"{slug(l)}__{a}__{vname}.npz").exists()]
        if not need:
            print(f"[skip] {vname} cached")
            continue
        t0 = time.time()
        labs = sorted({l for l, _ in need}, key=list(M).index)
        ds = VariantDS([SimpleNamespace(**r) for r in df.to_dict("records")], vname, [M[l]["pre"] for l in labs])
        feats = {k: [] for k in need}
        for batch in tqdm(DataLoader(ds, batch_size=BATCH, num_workers=workers, pin_memory=(device == "cuda")), desc=vname, leave=False):
            for lab, x in zip(labs, batch):
                x = x.to(device, non_blocking=True)
                for a in [a for l, a in need if l == lab]:
                    M[lab]["act"].mode = a
                    with torch.no_grad():
                        f = M[lab]["model"].encode_image(x)
                    feats[(lab, a)].append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
        for (lab, a), v in feats.items():
            np.savez(cache / f"{slug(lab)}__{a}__{vname}.npz", sample_ids=df.sample_id.to_numpy(), feats=np.concatenate(v))
        timings[f"embed_{vname}_s"] = round(time.time() - t0, 1)
        print(f"[done] {vname}: n={len(df)} in {timings[f'embed_{vname}_s']} s", flush=True)

    # ---- validation: new SwitchAct features must equal the earlier forced-QuickGELU / plain-GELU caches ----
    checks = {}
    old = {("ViT-B/32", "quickgelu"): "full", ("ViT-L/14", "quickgelu"): "full", ("BioCLIP", "gelu"): "full",
           ("ViT-B/32", "gelu"): "plain_gelu", ("ViT-L/14", "gelu"): "plain_gelu"}
    z_new_ids = None
    for (lab, a), folder in old.items():
        zo = np.load(CACHE / folder / f"{slug(lab)}__crop_full.npz")
        zn = np.load(cache / f"{slug(lab)}__{a}__crop_full.npz")
        pos = {int(s): i for i, s in enumerate(zo["sample_ids"])}
        idx = [pos[int(s)] for s in zn["sample_ids"]]
        checks[f"{lab}/{a} vs earlier {folder} run"] = float(np.abs(zo["feats"][idx] - zn["feats"]).max())
    print("feature validation (max abs diff vs earlier runs):", json.dumps(checks, indent=1), flush=True)

    # ---- predictions ----
    t0 = time.time()
    spec = sample.set_index("sample_id")
    plan = [("exp1", f"crop_{lvl}", lvl, "neutral") for lvl in LEVELS]
    plan += [("exp2_baseline", "orig", "original", "neutral")]
    plan += [("exp3", "orig", "original", p) for p in PROMPTS]
    plan += [("exp2_mismatch", f"mismatch_{h}", h, "neutral") for h in HABITATS]
    plan += [("exp4", "bgonly", "bg_only_blur30", "neutral")]
    plan = [t for t in plan if t[1] in variants]
    recs = []
    for lab, m in M.items():
        for a in ACTS:
            for exp, vname, cond, prompt in plan:
                z = np.load(cache / f"{slug(lab)}__{a}__{vname}.npz")
                sims = torch.from_numpy(z["feats"]).to(device) @ text_emb(m, a, prompt, device).T
                conf, pred = torch.softmax(m["scale"] * sims, dim=-1).max(dim=-1)
                ids, pred_sp = z["sample_ids"], np.array(CLASSES)[pred.cpu().numpy()]
                truth = spec.loc[ids, "species"].to_numpy()
                recs.append(pd.DataFrame({
                    "model": lab, "activation": a, "experiment": exp, "condition": cond, "prompt": prompt,
                    "habitat": cond if exp == "exp2_mismatch" else "", "sample_id": ids, "species": truth,
                    "image_path": [str(IMAGES / f) for f in spec.loc[ids, "file_name"]],
                    "prediction": pred_sp, "confidence": conf.cpu().numpy(), "correct": pred_sp == truth}))
    allp = pd.concat(recs, ignore_index=True)
    allp.to_csv(out_dir / "all_predictions.csv", index=False)
    timings["predictions_s"] = round(time.time() - t0, 1)
    timings["total_s"] = round(time.time() - t_all, 1)
    meta = {"classes": CLASSES, "n_classes": len(CLASSES), "random_baseline_pct": round(100 / len(CLASSES), 2),
            "activations": ACTS, "native_activation": NATIVE, "activation_layers_patched": {l: m["n_patched"] for l, m in M.items()},
            "feature_validation_max_abs_diff": checks, "n_samples": len(sample), "n_bg_only": len(bg),
            "n_mismatch_per_model_activation": n_mm, "batch_size": BATCH, "timings": timings, "smoke": smoke,
            "hf_hub_cache": os.environ.get("HF_HUB_CACHE", "default")}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(allp)} rows -> {out_dir / 'all_predictions.csv'} | total {timings['total_s']} s")
    b = allp[(allp.experiment == "exp2_baseline")].groupby(["model", "activation"]).correct.mean().mul(100).round(1)
    print("baseline acc (original, neutral):\n", b.unstack().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)

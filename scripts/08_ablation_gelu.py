"""Sensitivity check (NOT part of the main results): effect of forcing QuickGELU on the two OpenAI CLIP models.

Main pipeline forces quick_gelu=True (what the 'openai' weights were trained with). This rebuilds the models the way
open_clip 4.x does by default (plain GELU, with a warning) and re-scores crop_full / crop_ultra_tight / original.
"""
import json

import numpy as np
import open_clip
import pandas as pd
import torch
from torch.utils.data import DataLoader
from types import SimpleNamespace

from common import DATA, RESULTS, SPECIES, class_name, set_seed
from models import PROMPTS
from run_experiments import VariantDS

if __name__ == "__main__":
    set_seed()
    sample = pd.read_csv(DATA / "sample.csv")
    rows = [SimpleNamespace(**r) for r in sample.to_dict("records")]
    truth = np.array(SPECIES)[[SPECIES.index(s) for s in sample.species]]
    res = []
    for label, name in (("ViT-B/32", "ViT-B-32"), ("ViT-L/14", "ViT-L-14")):
        model, _, pre = open_clip.create_model_and_transforms(name, pretrained="openai")  # default: plain GELU (warns)
        tok = open_clip.get_tokenizer(name)
        model.cuda().eval()
        with torch.no_grad():
            t = model.encode_text(tok([PROMPTS["neutral"].format(class_name(s)) for s in SPECIES]).cuda())
            t = t / t.norm(dim=-1, keepdim=True)
            for variant in ("crop_full", "crop_ultra_tight", "orig"):
                dl = DataLoader(VariantDS(rows, variant, [pre]), batch_size=64, num_workers=6)
                preds = []
                for (x,) in dl:
                    f = model.encode_image(x.cuda()); f = f / f.norm(dim=-1, keepdim=True)
                    preds.append((f @ t.T).argmax(-1).cpu().numpy())
                acc = float((np.array(SPECIES)[np.concatenate(preds)] == truth).mean() * 100)
                res.append(dict(model=label, variant=variant, acc_plain_gelu=round(acc, 2)))
                print(res[-1], flush=True)
    pd.DataFrame(res).to_csv(RESULTS / "ablation_plain_gelu.csv", index=False)

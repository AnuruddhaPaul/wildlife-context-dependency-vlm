"""Bootstrap 95% CIs (resampling samples, 2000x, seed 42) for the headline metrics -> results/bootstrap_ci.csv."""
import sys

import numpy as np
import pandas as pd

import paper_values as PV
from common import RESULTS, SEED
from models import PROMPTS

RESULTS = RESULTS / sys.argv[1] if len(sys.argv) > 1 else RESULTS  # optional sub-folder, e.g. animal_only/primary
P = pd.read_csv(RESULTS / "all_predictions.csv")
rng = np.random.default_rng(SEED)
B = 2000


def vec(m, exp, cond=None, prompt=None):
    q = P[(P.model == m) & (P.experiment == exp)]
    if cond is not None:
        q = q[q.condition == cond]
    if prompt is not None:
        q = q[q.prompt == prompt]
    return q.groupby("sample_id").correct.mean().sort_index()  # mean over the (<=2) predictions of one sample


rows = []
for m in PV.MODELS:
    full, ut = vec(m, "exp1", "full"), vec(m, "exp1", "ultra_tight")
    base, mm = vec(m, "exp2_baseline"), vec(m, "exp2_mismatch")
    pr = {p: vec(m, "exp3", "original", p) for p in PROMPTS}
    bg = vec(m, "exp4")
    n = len(full)
    draws = {k: [] for k in ("acc_full", "acc_ultra_tight", "CDS", "acc_baseline", "acc_mismatch", "CR", "PS", "BG_only")}
    for _ in range(B):
        i = rng.integers(0, n, n)
        a_f, a_u = full.values[i].mean() * 100, ut.values[i].mean() * 100
        a_b, a_m = base.values[i].mean() * 100, mm.values[i].mean() * 100
        accs = np.array([pr[p].values[i].mean() * 100 for p in PROMPTS])
        j = rng.integers(0, len(bg), len(bg))
        draws["acc_full"].append(a_f); draws["acc_ultra_tight"].append(a_u)
        draws["CDS"].append((a_f - a_u) / a_f * 100 if a_f else 0.0)
        draws["acc_baseline"].append(a_b); draws["acc_mismatch"].append(a_m); draws["CR"].append(a_m / a_b * 100)
        draws["PS"].append((accs.max() - accs.min()) / accs.mean() * 100); draws["BG_only"].append(bg.values[j].mean() * 100)
    point = {"acc_full": full.mean() * 100, "acc_ultra_tight": ut.mean() * 100, "acc_baseline": base.mean() * 100,
             "acc_mismatch": mm.mean() * 100, "BG_only": bg.mean() * 100}
    point["CDS"] = (point["acc_full"] - point["acc_ultra_tight"]) / point["acc_full"] * 100
    point["CR"] = point["acc_mismatch"] / point["acc_baseline"] * 100
    a = np.array([pr[p].mean() * 100 for p in PROMPTS]); point["PS"] = (a.max() - a.min()) / a.mean() * 100
    tgt = PV.TARGETS[m]
    paper = {"acc_full": tgt["acc_full"], "acc_ultra_tight": tgt["acc_ultra_tight"], "CDS": tgt["CDS"], "CR": tgt["CR"],
             "PS": tgt["PS"], "BG_only": tgt["BG"], "acc_baseline": PV.EXP2[m][0], "acc_mismatch": PV.EXP2[m][1]}
    for k, d in draws.items():
        lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append(dict(model=m, metric=k, ours=round(point[k], 1), ci_lo=round(lo, 1), ci_hi=round(hi, 1),
                         paper=paper[k], paper_inside_95ci=bool(lo <= paper[k] <= hi)))
out = pd.DataFrame(rows)
out.to_csv(RESULTS / "bootstrap_ci.csv", index=False)
print(out.to_string(index=False))

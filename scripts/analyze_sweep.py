"""Post-processing of the animal-only activation sweep. Usage: python analyze_sweep.py [--smoke]

Writes into results/animal_only[_smoke]/: activation_sweep.csv, crop_diagnostics.csv, vs_17class.csv, primary/ (full analysis of
the native-activation slice: QuickGELU for the OpenAI models, GELU for BioCLIP) and figures into figures/animal_only[_smoke]/.
"""
import argparse
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import analyze
import paper_values as PV
from analyze import acc, cds, ps
from common import DATA, FIGURES, RESULTS, SPECIES
from imaging import LEVELS
from models import PROMPTS
from run_sweep import ACTS, NATIVE

MODELS = PV.MODELS
ACT_LABEL = {"quickgelu": "QuickGELU", "gelu": "GELU", "gelu_tanh": "GELU-tanh", "silu": "SiLU", "relu": "ReLU"}


def grid_row(P, model, act) -> dict:
    q = P[(P.model == model) & (P.activation == act)]

    def sel(exp, cond=None, prompt=None):
        d = q[q.experiment == exp]
        if cond is not None:
            d = d[d.condition == cond]
        if prompt is not None:
            d = d[d.prompt == prompt]
        return d

    r = dict(model=model, activation=act, native=NATIVE[model] == act)
    for lvl in LEVELS:
        r[f"acc_{lvl}"] = acc(sel("exp1", lvl))
    full, ut = sel("exp1", "full"), sel("exp1", "ultra_tight")
    r["CDS"] = cds(r["acc_full"], r["acc_ultra_tight"])
    base, mm, bg = sel("exp2_baseline"), sel("exp2_mismatch"), sel("exp4")
    r["acc_baseline_orig"], r["acc_mismatch"] = acc(base), acc(mm)
    r["CR"] = r["acc_mismatch"] / r["acc_baseline_orig"] * 100 if r["acc_baseline_orig"] else float("nan")
    pa = [acc(sel("exp3", "original", p)) for p in PROMPTS]
    for p, a in zip(PROMPTS, pa):
        r[f"prompt_{p}"] = a
    r["PS"] = ps(pa) if np.mean(pa) else float("nan")
    r["BG_only"] = acc(bg)
    # collapse check: a broken (e.g. ReLU) model may predict one class for everything; accuracy alone would hide that
    vc = base.prediction.value_counts(normalize=True)
    r["pred_classes_used_orig"], r["top_pred_share_pct_orig"], r["top_pred_class_orig"] = int(len(vc)), 100 * vc.iloc[0], vc.index[0]
    r["conf_full"], r["conf_ultra_tight"] = 100 * full.confidence.mean(), 100 * ut.confidence.mean()
    r["conf_acc_gap_ultra_tight"] = r["conf_ultra_tight"] - r["acc_ultra_tight"]
    sp = sorted(set(full.species))
    r["neg_cds_species"] = int(sum(cds(acc(full[full.species == s]), acc(ut[ut.species == s])) < 0 for s in sp))
    r["n_species"] = len(sp)
    return r


def main(smoke: bool):
    tag = "animal_only_smoke" if smoke else "animal_only"
    out, figs = RESULTS / tag, FIGURES / tag
    figs.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(out / "all_predictions.csv")
    meta = json.loads((out / "run_meta.json").read_text())

    G = pd.DataFrame([grid_row(P, m, a) for m in MODELS for a in ACTS])
    G.round(2).to_csv(out / "activation_sweep.csv", index=False)

    # ---- primary slice -> full standard analysis (tables, confusion matrices, panels, ...) ----
    prim = out / "primary"
    prim.mkdir(exist_ok=True)
    mask = np.zeros(len(P), bool)
    for m, a in NATIVE.items():
        mask |= ((P.model == m) & (P.activation == a)).to_numpy()
    P[mask].drop(columns="activation").to_csv(prim / "all_predictions.csv", index=False)
    pm = dict(meta, primary_activation=NATIVE, note="native activation per model: QuickGELU (OpenAI weights), GELU (BioCLIP config)")
    (prim / "run_meta.json").write_text(json.dumps(pm, indent=2))
    analyze.main(smoke=False, tag=f"{tag}/primary")

    # ---- crop diagnostics (why car, and partly deer, are problematic CDS probes) ----
    s = pd.read_csv(DATA / "sample.csv")

    def cover(r, m):
        L, T = max(0, r.x - m * r.w), max(0, r.y - m * r.h)
        R, B = min(r.json_w, r.x + (1 + m) * r.w), min(r.json_h, r.y + (1 + m) * r.h)
        return (R - L) * (B - T) / (r.json_w * r.json_h)

    for m_, name in ((0.0, "box"), (0.25, "m025"), (0.5, "m050"), (1.0, "m100")):
        s[f"cover_{name}"] = s.apply(lambda r: cover(r, m_), axis=1)
    D = s.groupby("species").agg(
        n=("sample_id", "count"), box_frac_of_image_median=("cover_box", "median"),
        full_crop_frac_of_image_median=("cover_m100", "median"),
        pct_full_crop_is_whole_image=("cover_m100", lambda v: 100 * (v > 0.99).mean()),
        pct_loose_crop_is_whole_image=("cover_m050", lambda v: 100 * (v > 0.99).mean()),
        pct_box_is_whole_image=("cover_box", lambda v: 100 * (v > 0.99).mean())).round(3)
    D.sort_values("box_frac_of_image_median", ascending=False).to_csv(out / "crop_diagnostics.csv")

    # ---- comparison with the earlier 17-class runs ----
    cols = ["acc_full", "acc_ultra_tight", "CDS", "acc_baseline_orig", "acc_mismatch", "CR", "PS", "BG_only"]
    s17q = pd.read_csv(RESULTS / "summary_table.csv").set_index("model")
    s17p = pd.read_csv(RESULTS / "plain_gelu" / "summary_table.csv").set_index("model")
    rows = []
    for m in MODELS:
        paper = {**PV.TARGETS[m], "BG_only": PV.TARGETS[m]["BG"], "acc_baseline_orig": PV.EXP2[m][0], "acc_mismatch": PV.EXP2[m][1]}
        for c in cols:
            rows.append(dict(
                model=m, metric=c, paper_17class=paper.get(c, np.nan),
                c17_quickgelu_forced=s17q.loc[m, c], c17_plain_gelu=s17p.loc[m, c],
                animal16_quickgelu=G[(G.model == m) & (G.activation == "quickgelu")][c].iloc[0],
                animal16_gelu=G[(G.model == m) & (G.activation == "gelu")][c].iloc[0]))
    pd.DataFrame(rows).round(2).to_csv(out / "vs_17class.csv", index=False)

    # ---- figures ----
    sns.set_theme(style="whitegrid", context="talk")
    metrics = ["acc_full", "acc_ultra_tight", "CDS", "CR", "PS", "BG_only"]
    fig, axes = plt.subplots(2, 3, figsize=(24, 11))
    for ax, c in zip(axes.ravel(), metrics):
        W = G.pivot(index="model", columns="activation", values=c).loc[MODELS, ACTS]
        sns.heatmap(W.rename(columns=ACT_LABEL), annot=True, fmt=".1f", cmap="viridis", cbar=False, ax=ax, annot_kws={"size": 15})
        ax.set_title(c); ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("Activation sweep, animal-only (16 classes). Native: QuickGELU (OpenAI), GELU (BioCLIP)", fontsize=18)
    fig.tight_layout(); fig.savefig(figs / "fig6_activation_sweep_heatmaps.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
    lv = list(LEVELS)
    for ax, m in zip(axes, MODELS):
        for a in ACTS:
            r = G[(G.model == m) & (G.activation == a)].iloc[0]
            ax.plot(lv, [r[f"acc_{l}"] for l in lv], "o-" if r.native else "o--", lw=4 if r.native else 2, label=ACT_LABEL[a] + (" (native)" if r.native else ""))
        ax.set_title(m); ax.set_xlabel("context level"); ax.legend(fontsize=12)
    axes[0].set_ylabel("Accuracy (%)")
    fig.tight_layout(); fig.savefig(figs / "fig7_activation_accuracy_vs_context.png", dpi=170); plt.close(fig)

    print(G[["model", "activation", "native", "acc_full", "acc_ultra_tight", "CDS", "acc_baseline_orig", "acc_mismatch", "CR", "PS", "BG_only",
             "pred_classes_used_orig", "top_pred_share_pct_orig", "top_pred_class_orig"]].round(1).to_string(index=False))
    print(D.sort_values("box_frac_of_image_median", ascending=False).head(5).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)

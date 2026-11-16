"""Metrics, tables, figures and example panels from all_predictions.csv. Usage: python analyze.py [--smoke]"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

import paper_values as PV
from common import DATA, FIGURES, RESULTS, SPECIES, class_name
from habitats import HABITAT_MAP, HABITATS, habitats_for
from imaging import LEVELS, load_image, make_variant, scaled_box
from models import PROMPTS

MODELS = PV.MODELS
COLORS = {"ViT-B/32": "#4c72b0", "ViT-L/14": "#dd8452", "BioCLIP": "#55a868"}
sns.set_theme(style="whitegrid", context="talk")


def acc(df) -> float:
    return 100.0 * df.correct.mean() if len(df) else float("nan")


def cds(a_full, a_ut) -> float:
    return 0.0 if a_full == 0 else (a_full - a_ut) / a_full * 100.0  # defined as 0 when Acc_full == 0


def ps(accs, how="mean") -> float:
    accs = np.asarray(accs, dtype=float)
    den = {"mean": accs.mean(), "max": accs.max(), "min": accs.min()}[how]
    return (accs.max() - accs.min()) / den * 100.0


def main(smoke: bool, plain_gelu: bool = False, tag: str | None = None):
    out = RESULTS / tag if tag else (RESULTS / "smoke" if smoke else (RESULTS / "plain_gelu" if plain_gelu else RESULTS))
    fig_dir = FIGURES / tag if tag else (FIGURES / "smoke" if smoke else (FIGURES / "plain_gelu" if plain_gelu else FIGURES))
    fig_dir.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(out / "all_predictions.csv")
    meta = json.loads((out / "run_meta.json").read_text())
    env = json.loads((RESULTS / "env_info.json").read_text())
    sample = pd.read_csv(DATA / "sample.csv").set_index("sample_id")
    species_here = [s for s in SPECIES if s in set(P.species)]

    def sel(model, exp, cond=None, prompt=None):
        q = P[(P.model == model) & (P.experiment == exp)]
        if cond is not None:
            q = q[q.condition == cond]
        if prompt is not None:
            q = q[q.prompt == prompt]
        return q

    # ---------------- tables ----------------
    level_rows, summary, hab_rows, prompt_rows, sp_rows = [], [], [], [], []
    for m in MODELS:
        for lvl in LEVELS:
            q = sel(m, "exp1", lvl)
            level_rows.append(dict(model=m, level=lvl, margin=LEVELS[lvl], acc=acc(q), mean_conf=100 * q.confidence.mean(), n=len(q)))
        full, ut = sel(m, "exp1", "full"), sel(m, "exp1", "ultra_tight")
        a_full, a_ut = acc(full), acc(ut)
        base, mm, bgo = sel(m, "exp2_baseline"), sel(m, "exp2_mismatch"), sel(m, "exp4")
        pacc = {p: acc(sel(m, "exp3", "original", p)) for p in PROMPTS}
        v = list(pacc.values())
        summary.append(dict(
            model=m, acc_full=a_full, acc_ultra_tight=a_ut, CDS=cds(a_full, a_ut), acc_baseline_orig=acc(base),
            acc_mismatch=acc(mm), CR=acc(mm) / acc(base) * 100, n_mismatch=len(mm),
            PS=ps(v), PS_denominator_max=ps(v, "max"), PS_denominator_min=ps(v, "min"),
            BG_only=acc(bgo), n_bg=len(bgo), conf_full=100 * full.confidence.mean(), conf_ultra_tight=100 * ut.confidence.mean(),
            conf_acc_gap_ultra_tight=100 * ut.confidence.mean() - a_ut,
            neg_cds_species=int(sum(cds(acc(full[full.species == s]), acc(ut[ut.species == s])) < 0 for s in species_here))))
        for h in HABITATS:
            q = sel(m, "exp2_mismatch", h)
            if q.empty:  # habitat unused in the smoke subset
                continue
            hab_rows.append(dict(model=m, habitat=h, correct=int(q.correct.sum()), total=len(q), acc=acc(q)))
        for p, a in pacc.items():
            prompt_rows.append(dict(model=m, prompt=p, template=PROMPTS[p], acc=a))
        for s in species_here:
            fs, us = full[full.species == s], ut[ut.species == s]
            sp_rows.append(dict(model=m, species=s, n=len(fs), acc_full=acc(fs), acc_ultra_tight=acc(us),
                                CDS=cds(acc(fs), acc(us)), acc_full_is_zero=bool(acc(fs) == 0)))
    S, L, H, PR, SP = map(pd.DataFrame, (summary, level_rows, hab_rows, prompt_rows, sp_rows))
    S.round(2).to_csv(out / "summary_table.csv", index=False)
    L.round(2).to_csv(out / "level_table.csv", index=False)
    H.round(2).to_csv(out / "habitat_table.csv", index=False)
    PR.round(2).to_csv(out / "prompt_table.csv", index=False)
    SPw = SP.pivot(index="species", columns="model", values="CDS")[MODELS]
    SPw.insert(0, "n", SP[SP.model == "ViT-B/32"].set_index("species").n)
    SPw.round(1).sort_values("BioCLIP", ascending=False).to_csv(out / "per_species_cds.csv")
    SP.round(2).to_csv(out / "per_species_long.csv", index=False)
    pd.DataFrame([dict(species=s, habitat_1=h[0], habitat_2=h[1], rationale=h[2]) for s, h in HABITAT_MAP.items()]).to_csv(
        out / "habitat_mapping.csv", index=False)

    # ---------------- comparison with the paper ----------------
    cmp = []
    for m in MODELS:
        row = S[S.model == m].iloc[0]
        for k, tv in PV.TARGETS[m].items():
            ours = row["BG_only" if k == "BG" else k]
            d = ours - tv
            cmp.append(dict(model=m, metric=k, paper=tv, ours=round(ours, 2), diff_pp=round(d, 2),
                            verdict="within 2pp" if abs(d) <= 2 else ("within 3pp" if abs(d) <= 3 else "DIFFERS >3pp")))
        cmp.append(dict(model=m, metric="conf-acc gap @ultra_tight", paper=PV.CONF_ACC_GAP_UT[m],
                        ours=round(row.conf_acc_gap_ultra_tight, 2), diff_pp=round(row.conf_acc_gap_ultra_tight - PV.CONF_ACC_GAP_UT[m], 2), verdict=""))
        for h, tv in PV.HABITAT_ACC[m].items():
            o = H[(H.model == m) & (H.habitat == h)].acc
            o = o.iloc[0] if len(o) else float("nan")
            cmp.append(dict(model=m, metric=f"mismatch acc {h}", paper=tv, ours=round(o, 2), diff_pp=round(o - tv, 2), verdict=""))
        for p, tv in PV.PROMPT_ACC[m].items():
            o = PR[(PR.model == m) & (PR.prompt == p)].acc.iloc[0]
            cmp.append(dict(model=m, metric=f"prompt acc {p}", paper=tv, ours=round(o, 2), diff_pp=round(o - tv, 2), verdict=""))
        cmp.append(dict(model=m, metric="Exp2 baseline acc", paper=PV.EXP2[m][0], ours=round(row.acc_baseline_orig, 2), diff_pp=round(row.acc_baseline_orig - PV.EXP2[m][0], 2), verdict=""))
        cmp.append(dict(model=m, metric="Exp2 mismatch acc", paper=PV.EXP2[m][1], ours=round(row.acc_mismatch, 2), diff_pp=round(row.acc_mismatch - PV.EXP2[m][1], 2), verdict=""))
    cmp.append(dict(model="ViT-B/32", metric="medium-crop acc", paper=38.7, ours=round(L[(L.model == "ViT-B/32") & (L.level == "medium")].acc.iloc[0], 2),
                    diff_pp=None, verdict=""))
    CMP = pd.DataFrame(cmp)
    CMP["diff_pp"] = CMP.ours - CMP.paper
    CMP.round(2).to_csv(out / "paper_comparison.csv", index=False)
    # PS formula check on the paper's OWN printed Table 4 accuracies
    ps_check = [dict(model=m, PS_reported=PV.TARGETS[m]["PS"],
                     PS_recomputed_from_paper_table4_mean_denominator=round(ps(list(PV.PROMPT_ACC[m].values())), 2),
                     PS_recomputed_max_denominator=round(ps(list(PV.PROMPT_ACC[m].values()), "max"), 2),
                     PS_recomputed_min_denominator=round(ps(list(PV.PROMPT_ACC[m].values()), "min"), 2),
                     PS_from_our_accuracies=round(S[S.model == m].PS.iloc[0], 2)) for m in MODELS]
    pd.DataFrame(ps_check).to_csv(out / "ps_formula_check.csv", index=False)
    spc = SPw[MODELS].copy()
    for i, m in enumerate(MODELS):
        spc[f"paper_{m}"] = [PV.SPECIES_CDS[s][i] for s in spc.index]
    spc.round(1).to_csv(out / "per_species_cds_vs_paper.csv")

    # ---------------- figures ----------------
    lv = list(LEVELS)
    fig, ax = plt.subplots(2, 1, figsize=(10, 11), sharex=True)
    for m in MODELS:
        q = L[L.model == m].set_index("level").loc[lv]
        ax[0].plot(lv, q.acc, "o-", lw=3, ms=9, color=COLORS[m], label=m)
        ax[1].plot(lv, q.mean_conf, "s--", lw=3, ms=9, color=COLORS[m], label=m)
    ax[0].set_ylabel("Accuracy (%)"); ax[1].set_ylabel("Mean confidence (%)")
    ax[1].set_xlabel("Context level (margin around bbox: 100% -> 0%)")
    ax[0].set_title("Accuracy vs. context level"); ax[1].set_title("Confidence vs. context level"); ax[0].legend()
    fig.tight_layout(); fig.savefig(fig_dir / "fig1_accuracy_confidence_vs_context.png", dpi=200); plt.close(fig)

    order = SPw.sort_values("BioCLIP", ascending=False).index
    fig, ax = plt.subplots(figsize=(16, 7))
    w = 0.27
    for i, m in enumerate(MODELS):
        ax.bar(np.arange(len(order)) + (i - 1) * w, SPw.loc[order, m], w, label=m, color=COLORS[m])
    ax.axhline(0, color="k", lw=1); ax.set_xticks(range(len(order)))
    ax.set_xticklabels([class_name(s) for s in order], rotation=45, ha="right")
    ax.set_ylabel("CDS (%)"); ax.set_title("Per-species Context Dependency Score (0 where Acc_full = 0)"); ax.legend()
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_per_species_cds.png", dpi=200); plt.close(fig)

    for tag, exp, cond in (("full_context_m1.0", "exp1", "full"), ("original_uncropped", "exp2_baseline", "original")):
        fig, axes = plt.subplots(1, 3, figsize=(30, 9))
        for ax, m in zip(axes, MODELS):
            q = sel(m, exp, cond, "neutral")
            labs = [s for s in SPECIES if s in species_here]
            cm = confusion_matrix(q.species, q.prediction, labels=labs, normalize="true")
            sns.heatmap(cm, ax=ax, cmap="Blues", vmin=0, vmax=1, cbar=False, xticklabels=[class_name(s) for s in labs],
                        yticklabels=[class_name(s) for s in labs], annot=len(labs) <= 6, fmt=".2f")
            ax.set_title(f"{m} (acc {acc(q):.1f}%)"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            single, ax1 = plt.subplots(figsize=(11, 9.5))
            sns.heatmap(cm, ax=ax1, cmap="Blues", vmin=0, vmax=1, xticklabels=[class_name(s) for s in labs],
                        yticklabels=[class_name(s) for s in labs])
            ax1.set_title(f"{m} - {tag} (row-normalised)"); ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
            single.tight_layout(); single.savefig(fig_dir / f"fig3_confusion_{tag}_{m.replace('/', '')}.png", dpi=170); plt.close(single)
        fig.tight_layout(); fig.savefig(fig_dir / f"fig3_confusion_{tag}_all_models.png", dpi=130); plt.close(fig)

    metrics = ["acc_full", "acc_ultra_tight", "CDS", "acc_baseline_orig", "acc_mismatch", "CR", "PS", "BG_only"]
    M = S.set_index("model")[metrics].loc[MODELS]
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap((M - M.min()) / (M.max() - M.min() + 1e-9), annot=M.round(1), fmt="", cmap="YlGnBu", cbar=False, ax=ax,
                linewidths=1, annot_kws={"size": 16})
    ax.set_title("Model comparison (cell colour = column-wise min-max scaling)"); ax.set_ylabel("")
    fig.tight_layout(); fig.savefig(fig_dir / "fig4_model_comparison_heatmap.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    w = 0.2
    for i, p in enumerate(PROMPTS):
        vals = [PR[(PR.model == m) & (PR.prompt == p)].acc.iloc[0] for m in MODELS]
        ax.bar(np.arange(3) + (i - 1.5) * w, vals, w, label=p)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{m}\nPS={S[S.model == m].PS.iloc[0]:.1f}%" for m in MODELS])
    ax.set_ylabel("Accuracy on original images (%)"); ax.set_title("Prompt sensitivity"); ax.legend(title="prompt")
    fig.tight_layout(); fig.savefig(fig_dir / "fig5_prompt_sensitivity.png", dpi=200); plt.close(fig)

    # ---------------- qualitative panels ----------------
    def preds(sid, exp, cond):
        q = P[(P.sample_id == sid) & (P.experiment == exp) & (P.condition == cond) & (P.prompt == "neutral")]
        return " | ".join(f"{r.model}: {r.prediction}" for r in q.itertuples())

    def first(sp, exp):
        q = P[(P.species == sp) & (P.experiment == exp)]
        return None if q.empty else int(q.sample_id.min())

    def render(sid, variant):
        r = sample.loc[sid]
        img = load_image(r.file_name)
        box = scaled_box(img, r.x, r.y, r.w, r.h, r.json_w, r.json_h)
        return make_variant(img, box, variant, sid)

    def panel(name, sid, items):
        fig, axes = plt.subplots(1, len(items), figsize=(7 * len(items), 6.3))
        axes = np.atleast_1d(axes)
        for ax, (variant, exp, cond, title) in zip(axes, items):
            ax.imshow(render(sid, variant)); ax.axis("off")
            ax.set_title(f"{title}\n{preds(sid, exp, cond)}", fontsize=11)
        fig.suptitle(f"{name}: true = {sample.loc[sid].species} (sample {sid}, {sample.loc[sid].file_name})", fontsize=13)
        fig.tight_layout(); fig.savefig(fig_dir / f"panel_{name}.png", dpi=170); plt.close(fig)

    ex_dir = fig_dir
    for sp in ("dog", "opossum"):
        sid = first(sp, "exp1")
        if sid is not None:
            panel(f"{sp}_full_vs_ultratight", sid, [("crop_full", "exp1", "full", "full context (margin 100%)"),
                                                    ("crop_ultra_tight", "exp1", "ultra_tight", "ultra-tight (0%)")])
    sid = first("raccoon", "exp2_mismatch")
    if sid is not None:
        h1, h2 = habitats_for("raccoon")
        panel("raccoon_mismatched_backgrounds", sid, [("orig", "exp2_baseline", "original", "original"),
                                                      (f"mismatch_{h1}", "exp2_mismatch", h1, f"{h1} background"),
                                                      (f"mismatch_{h2}", "exp2_mismatch", h2, f"{h2} background")])
    for sp in ("deer", "badger", "car"):
        sid = first(sp, "exp4")
        if sid is not None:
            panel(f"{sp}_background_only", sid, [("orig", "exp2_baseline", "original", "original"),
                                                 ("bgonly", "exp4", "bg_only_blur30", "background only (animal blurred, r=30)")])

    # ---------------- JSON ----------------
    detailed = {"env": env, "run_meta": meta, "seed": 42,
                "sample_counts": P.drop_duplicates("sample_id").species.value_counts().to_dict(),
                "n_classes": len(species_here) if tag else 17,
                "sampling_rule": "per-box sampling (each bbox annotation = 1 sample), up to 100 per species, numpy default_rng(42)",
                "habitat_mapping": {s: list(h[:2]) for s, h in HABITAT_MAP.items()},
                "summary": S.round(3).to_dict("records"), "levels": L.round(3).to_dict("records"),
                "habitat": H.round(3).to_dict("records"), "prompts": PR.round(3).to_dict("records"),
                "per_species": SP.round(3).to_dict("records"), "ps_formula_check": ps_check,
                "paper_comparison": CMP.round(3).where(CMP.notna(), None).to_dict("records")}
    (out / "detailed_results.json").write_text(json.dumps(detailed, indent=2, default=str))
    print(S.round(1).to_string(index=False))
    print(CMP[CMP.metric.isin(list(PV.TARGETS["BioCLIP"]))].to_string(index=False))
    print(pd.DataFrame(ps_check).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--plain-gelu", action="store_true")
    ap.add_argument("--tag", default=None, help="results sub-folder holding all_predictions.csv + run_meta.json")
    a = ap.parse_args()
    main(a.smoke, a.plain_gelu, a.tag)

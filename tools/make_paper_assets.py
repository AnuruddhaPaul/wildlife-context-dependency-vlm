"""Generate the paper's figures (PDF/PNG), LaTeX tables, CSV copies and numbers.tex macros from the saved results.

Everything quoted in the paper text comes from numbers.tex (generated here), so text and tables cannot drift from the data.
"""
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from common import class_name  # noqa: E402
from habitats import HABITAT_MAP  # noqa: E402
from imaging import load_image, make_variant, scaled_box  # noqa: E402
from nbhelpers import HABS, LEVELS, MODEL_COLORS, MODELS, PROMPT_KEYS  # noqa: E402

NA, NR = ROOT / "results" / "notebook_animal", ROOT / "results" / "notebook_report"
PAPER = ROOT / "paper"
FIGS, TABS, TCSV = PAPER / "figures", PAPER / "tables", PAPER / "tables_csv"
for d in (FIGS, TABS, TCSV):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6.5, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
                     "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})
sns.set_style("whitegrid", {"grid.linewidth": 0.4})
W2, W1 = 7.16, 3.5                      # IEEE double / single column width (inches)
ACTS = ["quickgelu", "gelu"]
ALAB = {"quickgelu": "QuickGELU", "gelu": "GELU"}
MK = {"ViT-B/32": "VB", "ViT-L/14": "VL", "BioCLIP": "Bio"}
SHORT = {"ViT-B/32": "B/32", "ViT-L/14": "L/14", "BioCLIP": "Bio"}

# ------------------------------------------------------------------ load
SUM = {a: pd.read_csv(NA / f"summary_{a}.csv").set_index("model") for a in ACTS}
LEV = {a: pd.read_csv(NA / f"levels_{a}.csv", index_col=0) for a in ACTS}
HAB = {a: pd.read_csv(NA / f"habitat_{a}.csv", index_col=0) for a in ACTS}
PRM = {a: pd.read_csv(NA / f"prompts_{a}.csv", index_col=0) for a in ACTS}
SPC = {a: pd.read_csv(NA / f"species_{a}.csv", index_col=0) for a in ACTS}
S17 = {r: pd.read_csv(NR / f"summary_{r}.csv").set_index("model") for r in ("17c-G", "17c-Q")}
CI = pd.read_csv(NR / "table02_bootstrap_ci.csv").query("run == '16c-Q'")
DN = pd.read_csv(NR / "eda_day_vs_night.csv")
RHO = pd.read_csv(NR / "eda_cds_size_correlation.csv").set_index("model")
CAL = pd.read_csv(NR / "eda_calibration.csv")
S16 = pd.read_csv(NA / "sample_points.csv")
CROP = pd.read_csv(ROOT / "results" / "animal_only" / "crop_diagnostics.csv").set_index("species")
META = json.loads((NA / "run_meta.json").read_text())
PRED = pd.read_csv(NA / "all_predictions.csv", usecols=["model", "activation", "experiment", "condition", "prompt", "sample_id", "species", "prediction", "correct"])

# ------------------------------------------------------------------ numbers.tex
def tx(v, d=1):
    v = round(float(v), d)
    return f"$-${abs(v):.{d}f}" if v < 0 else f"{v:.{d}f}"

MAC = {}
def mac(name, val):
    assert name.isalpha(), name
    MAC[name] = val

METRIC = {"acc_full": "Full", "acc_ultra_tight": "UT", "CDS": "CDS", "acc_baseline": "Base", "acc_baseline_orig": "Base", "acc_mismatch": "Mism",
          "CR": "CR", "PS": "PS", "BG_only": "BG", "conf_acc_gap_ut": "Gap", "conf_full": "ConfFull", "conf_ultra_tight": "ConfUT"}
for run, df in [("Q", SUM["quickgelu"]), ("G", SUM["gelu"]), ("SG", S17["17c-G"]), ("SQ", S17["17c-Q"])]:
    for m in MODELS:
        for col, nm in METRIC.items():
            if col in df.columns:
                mac(f"{run}{MK[m]}{nm}", tx(df.loc[m, col]))
        mac(f"{run}{MK[m]}Neg", str(int(df.loc[m, "neg_cds_species"])))
        mac(f"{run}{MK[m]}Bgn", str(int(round(df.loc[m, "BG_only"] / 100 * df.loc[m, "n_bg"]))))
mac("NBg", str(int(SUM["quickgelu"].n_bg.iloc[0])))
mac("NMismatch", f"{int(SUM['quickgelu'].n_mismatch.iloc[0]):,}".replace(",", "{,}"))
for run, a in (("Q", "quickgelu"), ("G", "gelu")):
    for m in MODELS:
        L = LEV[a].query("model == @m").set_index("level")
        for lv, nm in zip(LEVELS, ["Full", "Loose", "Medium", "Tight", "UT"]):
            mac(f"{run}{MK[m]}Lv{nm}", tx(L.loc[lv, "acc"])); mac(f"{run}{MK[m]}Lc{nm}", tx(L.loc[lv, "conf"]))
        for h in HABS:
            mac(f"{run}{MK[m]}Hab{h.capitalize()}", tx(HAB[a].query("model == @m and habitat == @h").acc.iloc[0]))
        for p in PROMPT_KEYS:
            mac(f"{run}{MK[m]}Prm{p.capitalize()}", tx(PRM[a].query("model == @m and prompt == @p").acc.iloc[0]))
        pr = PRM[a].query("model == @m").set_index("prompt").acc
        mac(f"{run}{MK[m]}PrmWild", tx(pr["wildlife"] - pr["neutral"])); mac(f"{run}{MK[m]}PrmHab", tx(pr["habitat"] - pr["neutral"]))
cmap = {"acc_full": "Full", "acc_ultra_tight": "UT", "CDS": "CDS", "acc_baseline": "Base", "acc_mismatch": "Mism", "CR": "CR", "PS": "PS", "BG_only": "BG"}
for _, r in CI.iterrows():
    k = f"Q{MK[r.model]}{cmap[r.metric]}"
    mac(k + "Lo", tx(r.ci_lo)); mac(k + "Hi", tx(r.ci_hi))
dFull, dPS, dCDS = {}, {}, {}
for m in MODELS:
    dFull[m] = SUM["quickgelu"].loc[m, "acc_full"] - SUM["gelu"].loc[m, "acc_full"]
    dPS[m] = SUM["quickgelu"].loc[m, "PS"] - SUM["gelu"].loc[m, "PS"]
    dCDS[m] = SUM["quickgelu"].loc[m, "CDS"] - SUM["gelu"].loc[m, "CDS"]
    mac(f"Delta{MK[m]}Full", tx(dFull[m])); mac(f"Delta{MK[m]}PS", tx(dPS[m])); mac(f"Delta{MK[m]}CDS", tx(dCDS[m]))
mac("MaxDeltaFull", tx(max(abs(v) for v in dFull.values()))); mac("MaxDeltaPS", tx(max(abs(v) for v in dPS.values())))
for _, r in DN.iterrows():
    tag = "Day" if r.frames.startswith("colour") else "Night"
    mac(f"{tag}{MK[r.model]}CDS", tx(r.CDS)); mac(f"{tag}{MK[r.model]}Full", tx(r.acc_full)); mac(f"{tag}{MK[r.model]}UT", tx(r.acc_ut)); mac(f"{tag}N", str(int(r.n)))
mac("NightShare", tx(100 * S16.is_gray_ir.mean(), 0))
for m in MODELS:
    mac(f"Rho{MK[m]}", tx(RHO.loc[m, "spearman_rho"], 2)); mac(f"Pval{MK[m]}", tx(RHO.loc[m, "p_value"], 2))
    for ctx, nm in (("full", "Full"), ("ultra-tight", "UT")):
        c = CAL.query("model == @m and context == @ctx").iloc[0]
        mac(f"Ece{nm}{MK[m]}", tx(c.ECE)); mac(f"CalConf{nm}{MK[m]}", tx(c.mean_conf))
    p = META["params"][m]; mac(f"Params{MK[m]}", tx(p["params_M"])); mac(f"Patch{MK[m]}", str(p["patch"]))
mac("NSamples", f"{len(S16):,}".replace(",", "{,}")); mac("NImages", f"{S16.file_name.nunique():,}".replace(",", "{,}"))
mac("NRepeatImages", str(int((S16.file_name.value_counts() > 1).sum())))
mac("NCameras", str(S16.location.nunique()))
mac("CarCrop", tx(CROP.loc["car", "pct_full_crop_is_whole_image"], 0)); mac("DeerCrop", tx(CROP.loc["deer", "pct_full_crop_is_whole_image"], 0))
tm = META["timings"]; mac("RuntimeMin", tx(tm["experiments_embedding_total_s"] / 60)); mac("GPUName", META["env"].get("gpu", "GPU"))
mac("TorchVer", META["env"]["torch"].replace("+", "{+}")); mac("OpenCLIPVer", META["env"]["open_clip"])
(PAPER / "numbers.tex").write_text("% AUTO-GENERATED by tools/make_paper_assets.py - do not edit\n" +
                                   "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MAC.items())) + "\n", encoding="utf-8")
print(len(MAC), "macros written")

# ------------------------------------------------------------------ figures
def save(fig, name, png=False):
    fig.savefig(FIGS / f"{name}.pdf"); fig.savefig(FIGS / f"{name}.png", dpi=300 if png else 220); plt.close(fig)

# F1: experimental conditions
r = S16[S16.species == "dog"].iloc[0]
img = load_image(r.file_name); box = scaled_box(img, r.x, r.y, r.w, r.h, r.json_w, r.json_h)
h1, h2 = HABITAT_MAP["dog"][:2]
panels = [("crop_full", "Full (100%)"), ("crop_loose", "Loose (50%)"), ("crop_medium", "Medium (25%)"), ("crop_tight", "Tight (5%)"),
          ("crop_ultra_tight", "Ultra-tight (0%)"), (f"mismatch_{h1}", f"Mismatch: {h1}"), (f"mismatch_{h2}", f"Mismatch: {h2}"), ("bgonly", "Background only")]
fig, axs = plt.subplots(2, 4, figsize=(W2, 3.5))
for ax, (v, t) in zip(axs.ravel(), panels):
    im = make_variant(img, box, v, int(r.sample_id)); im.thumbnail((640, 640))
    ax.imshow(im); ax.set_title(t, fontsize=7); ax.axis("off")
fig.tight_layout(pad=0.3); save(fig, "fig_conditions", png=True)

# F2: accuracy / confidence vs context
LL = ["Full", "Loose", "Medium", "Tight", "Ultra-\ntight"]
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.5))
for ax, key, yl in zip(axs, ["acc", "conf"], ["Accuracy (%)", "Mean confidence (%)"]):
    for m in MODELS:
        for a, ls, mk in (("quickgelu", "-", "o"), ("gelu", "--", "s")):
            q = LEV[a].query("model == @m").set_index("level").loc[LEVELS]
            ax.plot(range(5), q[key], ls, marker=mk, ms=3.5, color=MODEL_COLORS[m], label=f"{m}, {ALAB[a]}")
    ax.set_xticks(range(5)); ax.set_xticklabels(LL); ax.set_ylabel(yl); ax.set_xlabel("Context kept around the animal")
axs[0].set_title("(a) Accuracy"); axs[1].set_title("(b) Confidence"); axs[1].legend(ncol=1, loc="lower right", frameon=True, fontsize=6)
fig.tight_layout(); save(fig, "fig_context_curves")

# F3: per-species CDS
q = SPC["quickgelu"].pivot(index="species", columns="model", values="CDS")[MODELS]
g = SPC["gelu"].pivot(index="species", columns="model", values="CDS")[MODELS]
order = q.sort_values("BioCLIP", ascending=False).index
fig, ax = plt.subplots(figsize=(W2, 2.6)); w = 0.26; x = np.arange(len(order))
for i, m in enumerate(MODELS):
    ax.bar(x + (i - 1) * w, q.loc[order, m], w, color=MODEL_COLORS[m], label=f"{m} (QuickGELU)")
    ax.scatter(x + (i - 1) * w, g.loc[order, m], marker="D", s=9, facecolor="white", edgecolor="k", linewidth=0.6, zorder=5, label="GELU" if i == 0 else None)
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels([class_name(s) for s in order], rotation=40, ha="right")
ax.set_ylabel("CDS (%)"); ax.set_ylim(-230, 110); ax.legend(ncol=4, loc="lower left", frameon=True, fontsize=6)
fig.tight_layout(); save(fig, "fig_species_cds")

# F4: confusion matrices (full context, QuickGELU)
labs = sorted(PRED.species.unique())
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.55))
for ax, m in zip(axs, MODELS):
    d = PRED.query("model == @m and activation == 'quickgelu' and experiment == 'exp1' and condition == 'full' and prompt == 'neutral'")
    cm = confusion_matrix(d.species, d.prediction, labels=labs, normalize="true")
    sns.heatmap(cm, ax=ax, cmap="Blues", vmin=0, vmax=1, cbar=False, xticklabels=[class_name(s) for s in labs], yticklabels=[class_name(s) for s in labs], linewidths=0)
    ax.set_title(f"{m} ({100 * d.correct.mean():.1f}%)", fontsize=7); ax.set_xlabel("Predicted", fontsize=6.5); ax.tick_params(labelsize=4.3, length=1)
    if m != MODELS[0]: ax.set_yticklabels([]); ax.set_ylabel("")
    else: ax.set_ylabel("True", fontsize=6.5)
fig.tight_layout(w_pad=0.4); save(fig, "fig_confusion")

# F5: activation sensitivity
fig, axs = plt.subplots(1, 4, figsize=(W2, 2.0))
for ax, (col, ttl) in zip(axs, [("acc_full", "Acc$_{full}$ (%)"), ("CDS", "CDS (%)"), ("CR", "CR (%)"), ("PS", "PS (%)")]):
    for k, a in enumerate(ACTS):
        ax.bar(np.arange(3) + (k - 0.5) * 0.38, [SUM[a].loc[m, col] for m in MODELS], 0.38, label=ALAB[a], color=["#4c72b0", "#dd8452"][k])
    ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(3)); ax.set_xticklabels([SHORT[m] for m in MODELS]); ax.set_title(ttl)
axs[0].legend(loc="lower left", fontsize=6)
fig.tight_layout(); save(fig, "fig_activation")

# F6: day/night CDS + crop coverage
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw={"width_ratios": [1, 1.6]})
ax = axs[0]
for k, (frames, col, lab) in enumerate((("colour (day)", "#e1a53b", "Colour (day) frames"), ("grayscale / IR (night)", "#3b5b92", "Infrared (night) frames"))):
    ax.bar(np.arange(3) + (k - 0.5) * 0.38, [DN.query("model == @m and frames == @frames").CDS.iloc[0] for m in MODELS], 0.38, color=col, label=lab)
ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(3)); ax.set_xticklabels([SHORT[m] for m in MODELS]); ax.set_ylabel("CDS (%)")
ax.set_title("(a) CDS by frame type"); ax.legend(fontsize=6, loc="upper left")
ax = axs[1]
c = CROP.pct_full_crop_is_whole_image.sort_values(ascending=False)
ax.bar(range(len(c)), c.values, color=["#c44e52" if s == "car" else "#64b5cd" for s in c.index])
ax.set_xticks(range(len(c))); ax.set_xticklabels([class_name(s) for s in c.index], rotation=55, ha="right", fontsize=5.5)
ax.set_ylabel("Full crops = whole image (%)"); ax.set_title("(b) Where context removal cannot act (car = control)")
fig.tight_layout(); save(fig, "fig_daynight_coverage")

# F7: qualitative examples
SP = S16.set_index("sample_id")
PQ = PRED.query("activation == 'quickgelu' and prompt == 'neutral'")
def pred_txt(sid, exp, cond):
    d = PQ.query("sample_id == @sid and experiment == @exp and condition == @cond").set_index("model").prediction
    return "B/32: {} | L/14: {} | Bio: {}".format(*(d[m].replace("_", " ") for m in MODELS))
def first(sp, exp):
    return int(PQ.query("species == @sp and experiment == @exp").sample_id.min())
def draw(ax, sid, variant, title, exp, cond):
    rr = SP.loc[sid]; im0 = load_image(rr.file_name); b = scaled_box(im0, rr.x, rr.y, rr.w, rr.h, rr.json_w, rr.json_h)
    im = make_variant(im0, b, variant, sid); im.thumbnail((700, 700)); ax.imshow(im); ax.axis("off")
    ax.set_title(f"{title}\n{pred_txt(sid, exp, cond)}", fontsize=4.6, pad=2)
sd, so, sr, sdeer = first("dog", "exp1"), first("opossum", "exp1"), first("raccoon", "exp2_mismatch"), first("deer", "exp4")
fig, axs = plt.subplots(2, 4, figsize=(W2, 3.6))
draw(axs[0, 0], sd, "crop_full", "dog: full context", "exp1", "full"); draw(axs[0, 1], sd, "crop_ultra_tight", "dog: ultra-tight", "exp1", "ultra_tight")
draw(axs[0, 2], so, "crop_full", "opossum: full context", "exp1", "full"); draw(axs[0, 3], so, "crop_ultra_tight", "opossum: ultra-tight", "exp1", "ultra_tight")
ra, rb = HABITAT_MAP["raccoon"][:2]
draw(axs[1, 0], sr, "orig", "raccoon: original", "exp2_baseline", "original"); draw(axs[1, 1], sr, f"mismatch_{ra}", f"raccoon on {ra}", "exp2_mismatch", ra)
draw(axs[1, 2], sr, f"mismatch_{rb}", f"raccoon on {rb}", "exp2_mismatch", rb); draw(axs[1, 3], sdeer, "bgonly", "deer: background only", "exp4", "bg_only_blur30")
fig.tight_layout(pad=0.3, h_pad=0.8); save(fig, "fig_qualitative", png=True)
print("figures done:", sorted(p.name for p in FIGS.glob("*.pdf")))

# ------------------------------------------------------------------ tables
def tab(name, body): (TABS / f"{name}.tex").write_text(body, encoding="utf-8")
def n(v, d=1): return tx(v, d)
HDR = r"\toprule"

# T1 dataset
g = S16.groupby("species").agg(n=("sample_id", "count"), cams=("location", "nunique"), box=("box_frac", "median"), night=("is_gray_ir", "mean"),
                                 cover=("cover_full", lambda v: 100 * (v > 0.99).mean()))
rows = "\n".join(f"{class_name(s)} & {int(r.n)} & {int(r.cams)} & {100 * r.box:.1f} & {100 * r.night:.0f} & {r.cover:.0f} \\\\" for s, r in g.sort_values("n", ascending=False).iterrows())
tab("tab_dataset", "\\begin{table}[t]\n\\caption{Evaluation set (16 animal species, one sample = one annotated box). Cameras = distinct camera locations; Box = median box area as \\% of the image; Night = share of grayscale/infrared frames; Whole = \\% of full-context crops that already cover the entire image.}\n\\label{tab:dataset}\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lrrrrr}\n"
    + HDR + "\nSpecies & Samples & Cameras & Box & Night & Whole \\\\\n\\midrule\n" + rows + "\n\\midrule\n"
    + f"Total & {len(S16):,} & {S16.location.nunique()} & {100 * S16.box_frac.median():.1f} & {100 * S16.is_gray_ir.mean():.0f} & {100 * (S16.cover_full > 0.99).mean():.0f} \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")

# T2 main results
def mrow(a, m):
    s = SUM[a].loc[m]; return " & ".join([n(s.acc_full), n(s.acc_ultra_tight), n(s.CDS), n(s.CR, 0), n(s.PS), n(s.BG_only)])
body = ("\\begin{table*}[t]\n\\caption{Main results on 16 animal classes. Acc in \\%; CDS, CR and PS in \\%; BG = background-only accuracy ($n{=}" + str(int(SUM['quickgelu'].n_bg.iloc[0])) + "$, chance $6.25\\%$). QuickGELU is the primary configuration.}\n\\label{tab:main}\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
        "\\begin{tabular}{l rrrrrr rrrrrr}\n" + HDR + "\n & \\multicolumn{6}{c}{QuickGELU (primary)} & \\multicolumn{6}{c}{GELU}\\\\\n\\cmidrule(lr){2-7}\\cmidrule(lr){8-13}\n"
        "Model & Acc$_\\text{full}$ & Acc$_\\text{UT}$ & CDS & CR & PS & BG & Acc$_\\text{full}$ & Acc$_\\text{UT}$ & CDS & CR & PS & BG \\\\\n\\midrule\n"
        + "\n".join(f"{m} & {mrow('quickgelu', m)} & {mrow('gelu', m)} \\\\" for m in MODELS) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")
tab("tab_main", body)

# T3 mismatch + habitat
def hrow(a, m):
    s = SUM[a].loc[m]; hh = HAB[a].query("model == @m").set_index("habitat").acc
    return " & ".join([n(s.acc_baseline), n(s.acc_mismatch), n(s.CR, 0)] + [n(hh[h]) for h in HABS])
nh = HAB["quickgelu"].query("model == 'BioCLIP'").set_index("habitat").total
rows = []
for a in ACTS:
    rows += [f"{ALAB[a] if i == 0 else ''} & {m} & {hrow(a, m)} \\\\" for i, m in enumerate(MODELS)] + ["\\midrule"]
tab("tab_mismatch", "\\begin{table*}[t]\n\\caption{Experiment 2: context robustness. Baseline = original image; Mismatch = animal pasted on a wrong-habitat background (accuracy over all " + f"{int(SUM['quickgelu'].n_mismatch.iloc[0]):,}" + " composites); CR = Mismatch/Baseline. Right: accuracy per synthetic habitat (arctic $n{=}" + str(int(nh['arctic'])) + "$, desert " + str(int(nh['desert'])) + ", aquatic " + str(int(nh['aquatic'])) + ", urban " + str(int(nh['urban'])) + "). Habitat accuracies mix background and species composition (Sec.~\\ref{sec:limits}).}\n\\label{tab:mismatch}\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{ll rrr rrrr}\n"
    + HDR + "\nActivation & Model & Baseline & Mismatch & CR & Arctic & Desert & Aquatic & Urban \\\\\n\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# T4 prompts
rows = []
for a in ACTS:
    for i, m in enumerate(MODELS):
        pr = PRM[a].query("model == @m").set_index("prompt").acc
        rows.append(f"{ALAB[a] if i == 0 else ''} & {m} & " + " & ".join(n(pr[p]) for p in PROMPT_KEYS) + f" & {n(SUM[a].loc[m, 'PS'])} \\\\")
    rows.append("\\midrule")
tab("tab_prompts", "\\begin{table}[t]\n\\caption{Experiment 3: accuracy (\\%) on original images for four prompt templates, and Prompt Sensitivity PS $=(\\max-\\min)/\\text{mean}\\times100$.}\n\\label{tab:prompts}\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{ll rrrr r}\n"
    + HDR + "\nAct. & Model & Neutral & Habitat & Wildlife & Scient. & PS \\\\\n\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# T5 BG-only
rows = []
for m in MODELS:
    cq = CI.query("model == @m and metric == 'BG_only'").iloc[0]
    rows.append(f"{m} & {n(SUM['quickgelu'].loc[m, 'BG_only'])} ({int(round(SUM['quickgelu'].loc[m, 'BG_only'] / 100 * 80))}/80) & [{n(cq.ci_lo)}, {n(cq.ci_hi)}] & {n(SUM['gelu'].loc[m, 'BG_only'])} ({int(round(SUM['gelu'].loc[m, 'BG_only'] / 100 * 80))}/80) \\\\")
tab("tab_bg", "\\begin{table}[t]\n\\caption{Experiment 4: accuracy (\\%) from images whose animal box is blurred ($n{=}80$; chance $=6.25\\%$, i.e.\\ 5 of 80). Brackets: 95\\% bootstrap CI.}\n\\label{tab:bg}\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l rl r}\n"
    + HDR + "\nModel & QuickGELU & 95\\% CI & GELU \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# T6 per-species CDS
q = SPC["quickgelu"].pivot(index="species", columns="model", values="CDS")[MODELS]; g = SPC["gelu"].pivot(index="species", columns="model", values="CDS")[MODELS]
order = q.sort_values("BioCLIP", ascending=False).index
rows = "\n".join(f"{class_name(s)} & " + " & ".join(n(q.loc[s, m], 0) for m in MODELS) + " & " + " & ".join(n(g.loc[s, m], 0) for m in MODELS) + " \\\\" for s in order)
tab("tab_species", "\\begin{table}[t]\n\\caption{Per-species CDS (\\%), sorted by BioCLIP/QuickGELU; defined as 0 where Acc$_\\text{full}=0$. B = ViT-B/32, L = ViT-L/14, Bio = BioCLIP.}\n\\label{tab:species}\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{l rrr rrr}\n"
    + HDR + "\n & \\multicolumn{3}{c}{QuickGELU} & \\multicolumn{3}{c}{GELU}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nSpecies & B & L & Bio & B & L & Bio \\\\\n\\midrule\n" + rows + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# T7 day/night
rows = []
for m in MODELS:
    for i, (frames, lab) in enumerate((("colour (day)", "Colour"), ("grayscale / IR (night)", "Infrared"))):
        d = DN.query("model == @m and frames == @frames").iloc[0]
        rows.append(f"{m if i == 0 else ''} & {lab} & {int(d.n)} & {n(d.acc_full)} & {n(d.acc_ut)} & {n(d.CDS)} \\\\")
    rows.append("\\midrule")
tab("tab_daynight", "\\begin{table}[t]\n\\caption{Context dependence by frame type (QuickGELU, Experiment 1). Colour frames are daytime; grayscale frames are infrared night images.}\n\\label{tab:daynight}\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{ll r rrr}\n"
    + HDR + "\nModel & Frames & $n$ & Acc$_\\text{full}$ & Acc$_\\text{UT}$ & CDS \\\\\n\\midrule\n" + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

# T8 17-class vs 16-class
def cmp_row(m):
    cells = []
    for df in (S17["17c-G"], S17["17c-Q"], SUM["quickgelu"]):
        s = df.loc[m]; cells += [n(s.acc_full), n(s.CDS), n(s.CR, 0)]
    return " & ".join(cells)
tab("tab_17class", "\\begin{table*}[t]\n\\caption{Sensitivity to the label set and activation. 17 classes include the non-animal control `car'. 17c-GELU: plain-GELU build (the default of recent open\\_clip for the OpenAI weights); 17c-QG: QuickGELU for the OpenAI models, native GELU for BioCLIP; 16c-QG: our primary configuration. Samples are identical across rows except for the removed car boxes.}\n\\label{tab:17class}\n\\centering\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l rrr rrr rrr}\n"
    + HDR + "\n & \\multicolumn{3}{c}{17c-GELU} & \\multicolumn{3}{c}{17c-QG} & \\multicolumn{3}{c}{16c-QG (primary)}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\nModel & Acc$_\\text{full}$ & CDS & CR & Acc$_\\text{full}$ & CDS & CR & Acc$_\\text{full}$ & CDS & CR \\\\\n\\midrule\n"
    + "\n".join(f"{m} & {cmp_row(m)} \\\\" for m in MODELS) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n")

# ------------------------------------------------------------------ csv copies for the zip
for a in ACTS:
    for k in ("summary", "levels", "habitat", "prompts", "species", "species_prompt"):
        shutil.copy(NA / f"{k}_{a}.csv", TCSV / f"{k}_{a}.csv")
for f in ("table02_bootstrap_ci.csv", "eda_day_vs_night.csv", "eda_size_bins.csv", "eda_calibration.csv", "eda_cds_size_correlation.csv", "eda_top_confusions.csv",
          "table04_context_robustness.csv", "table06_prompts.csv", "table_ps_formula_check.csv", "claims_reproduction_check.csv", "table08_hypotheses.csv", "summary_17c-G.csv", "summary_17c-Q.csv"):
    if (NR / f).exists(): shutil.copy(NR / f, TCSV / f)
shutil.copy(NA / "sample_points.csv", TCSV / "sample_points_16class.csv"); shutil.copy(NA / "habitat_mapping.csv", TCSV / "habitat_mapping.csv")
shutil.copy(ROOT / "results" / "animal_only" / "activation_sweep.csv", TCSV / "activation_sweep_5_activations.csv")
shutil.copy(ROOT / "results" / "animal_only" / "crop_diagnostics.csv", TCSV / "crop_diagnostics.csv")
print("tables:", sorted(p.name for p in TABS.glob("*.tex")))

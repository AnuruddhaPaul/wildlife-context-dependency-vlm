"""Metric + styling helpers shared by the two notebooks (formulas exactly as in the paper / JK document)."""
import numpy as np
import pandas as pd

MODELS = ["ViT-B/32", "ViT-L/14", "BioCLIP"]
MODEL_COLORS = {"ViT-B/32": "#4c72b0", "ViT-L/14": "#dd8452", "BioCLIP": "#55a868"}
LEVELS = ["full", "loose", "medium", "tight", "ultra_tight"]
LEVEL_LABELS = ["Full", "Loose", "Medium", "Tight", "Ultra-tight"]
PROMPT_KEYS = ["neutral", "habitat", "wildlife", "scientific"]
HABS = ["arctic", "desert", "aquatic", "urban"]


def acc(d) -> float:
    return 100.0 * float(d.correct.mean()) if len(d) else float("nan")


def cds(a_full: float, a_ut: float) -> float:
    """CDS = (Acc_full - Acc_ultra_tight) / Acc_full * 100 ; defined as 0 when Acc_full == 0."""
    return 0.0 if a_full == 0 else (a_full - a_ut) / a_full * 100.0


def ps(accs) -> float:
    """PS = (max - min) / mean * 100 (the denominator that reproduces the paper's printed PS values)."""
    a = np.asarray(accs, dtype=float)
    return float((a.max() - a.min()) / a.mean() * 100.0) if a.mean() else float("nan")


def summarize(P: pd.DataFrame) -> dict:
    """All paper tables for ONE run. P = long predictions (model, experiment, condition, prompt, species, correct, confidence)."""
    summ, lev, hab, prm, spc, spp = [], [], [], [], [], []
    for m in MODELS:
        q = P[P.model == m]
        e1 = q[q.experiment == "exp1"]
        for l in LEVELS:
            d = e1[e1.condition == l]
            lev.append(dict(model=m, level=l, acc=acc(d), conf=100 * d.confidence.mean(), n=len(d)))
        full, ut = e1[e1.condition == "full"], e1[e1.condition == "ultra_tight"]
        base, mm, bg = q[q.experiment == "exp2_baseline"], q[q.experiment == "exp2_mismatch"], q[q.experiment == "exp4"]
        e3 = q[q.experiment == "exp3"]
        pacc = {k: acc(e3[e3.prompt == k]) for k in PROMPT_KEYS}
        for k, v in pacc.items():
            prm.append(dict(model=m, prompt=k, acc=v))
        for h in HABS:
            d = mm[mm.condition == h]
            if len(d):
                hab.append(dict(model=m, habitat=h, correct=int(d.correct.sum()), total=len(d), acc=acc(d)))
        sp = sorted(full.species.unique())
        negs = 0
        for s in sp:
            f, u = acc(full[full.species == s]), acc(ut[ut.species == s])
            spc.append(dict(model=m, species=s, n=int((full.species == s).sum()), acc_full=f, acc_ultra_tight=u,
                            CDS=cds(f, u), acc_full_is_zero=bool(f == 0),
                            conf_full=100 * full[full.species == s].confidence.mean(),
                            conf_ultra_tight=100 * ut[ut.species == s].confidence.mean()))
            negs += cds(f, u) < 0
            for k in PROMPT_KEYS:
                spp.append(dict(model=m, species=s, prompt=k, acc=acc(e3[(e3.prompt == k) & (e3.species == s)])))
        a_f, a_u = acc(full), acc(ut)
        summ.append(dict(model=m, acc_full=a_f, acc_ultra_tight=a_u, CDS=cds(a_f, a_u), acc_baseline=acc(base),
                         acc_mismatch=acc(mm), CR=acc(mm) / acc(base) * 100 if acc(base) else float("nan"),
                         n_mismatch=len(mm), PS=ps(list(pacc.values())), BG_only=acc(bg), n_bg=len(bg),
                         conf_full=100 * full.confidence.mean(), conf_ultra_tight=100 * ut.confidence.mean(),
                         conf_acc_gap_ut=100 * ut.confidence.mean() - a_u, neg_cds_species=int(negs), n_species=len(sp)))
    return dict(summary=pd.DataFrame(summ).set_index("model"), levels=pd.DataFrame(lev), habitat=pd.DataFrame(hab),
                prompts=pd.DataFrame(prm), species=pd.DataFrame(spc), species_prompt=pd.DataFrame(spp))


def cds_color(v) -> str:
    """JK Table 3 colour code: red > 30, orange 15-30, green 0-15, blue negative."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "#ffffff"
    return "#9ecae1" if v < 0 else "#a1d99b" if v < 15 else "#fdae6b" if v < 30 else "#fb6a4a"


def bootstrap_ci(P: pd.DataFrame, B: int = 2000, seed: int = 42) -> pd.DataFrame:
    """95% bootstrap CIs (resampling samples) for the headline metrics of one run."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in MODELS:
        q = P[P.model == m]

        def vec(exp, cond=None, prompt=None):
            d = q[q.experiment == exp]
            if cond is not None:
                d = d[d.condition == cond]
            if prompt is not None:
                d = d[d.prompt == prompt]
            return d.groupby("sample_id").correct.mean().sort_index().values * 100

        full, ut, base, mm = vec("exp1", "full"), vec("exp1", "ultra_tight"), vec("exp2_baseline"), vec("exp2_mismatch")
        pr = [vec("exp3", "original", k) for k in PROMPT_KEYS]
        bg = vec("exp4")
        n = len(full)
        dr = {k: [] for k in ("acc_full", "acc_ultra_tight", "CDS", "acc_baseline", "acc_mismatch", "CR", "PS", "BG_only")}
        for _ in range(B):
            i, j = rng.integers(0, n, n), rng.integers(0, len(bg), len(bg))
            f, u, b, mmv = full[i].mean(), ut[i].mean(), base[i].mean(), mm[i].mean()
            pa = [p[i].mean() for p in pr]
            dr["acc_full"].append(f); dr["acc_ultra_tight"].append(u); dr["CDS"].append(cds(f, u))
            dr["acc_baseline"].append(b); dr["acc_mismatch"].append(mmv); dr["CR"].append(mmv / b * 100 if b else np.nan)
            dr["PS"].append(ps(pa)); dr["BG_only"].append(bg[j].mean())
        pt = dict(acc_full=full.mean(), acc_ultra_tight=ut.mean(), acc_baseline=base.mean(), acc_mismatch=mm.mean(), BG_only=bg.mean())
        pt["CDS"] = cds(pt["acc_full"], pt["acc_ultra_tight"]); pt["CR"] = pt["acc_mismatch"] / pt["acc_baseline"] * 100
        pt["PS"] = ps([p.mean() for p in pr])
        for k, v in dr.items():
            lo, hi = np.nanpercentile(v, [2.5, 97.5])
            rows.append(dict(model=m, metric=k, value=pt[k], ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def ece(conf, correct, bins: int = 10):
    """Expected calibration error (%) and per-bin (mean confidence, accuracy, count)."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    out, e = [], 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append((conf[m].mean(), correct[m].mean(), int(m.sum())))
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return 100 * e, out

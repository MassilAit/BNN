# binarized_interactions_folder.py
# ------------------------------------------------------------
# Scans Result/*.json (binarized files), tests:
#   - clip×bs interaction (vs additive base)
#   - clip×lr interaction (vs additive base)
#   - bs×lr   interaction (vs additive base)       <-- NEW
# Also:
#   - Within-clip partial LRTs:  (bs|clip, lr|clip)
#   - Within-lr   partial LRTs:  (bs|lr,  clip|lr) <-- NEW
#   - Extracts main-effect ORs from the additive model
#   - Produces a per-file recommendation string
#
# Outputs:
#   out/binarized_interactions_summary.csv   (1 row per file)
#   out/binarized_within_clip_details.csv    (per file × clip level)
#   out/binarized_within_lr_details.csv      (per file × lr   level) <-- NEW
#
# Notes:
#   - Data format: "lr=..., bs=..., clip=...": [accuracy list]
#   - Same index across lists = same seed
#   - SUCCESS_THRESHOLD controls success=1 vs 0 (e.g., 100.0)
#   - This version is robust to string/float mixing in levels
# ------------------------------------------------------------

import os
import json
import math
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

# ---------------- Settings ----------------
RESULT_DIR = "Result"
OUT_DIR = "out"
SUCCESS_THRESHOLD = 100.0
# Practical-equivalence band for OR (optional note)
SESOI_OR_LOW, SESOI_OR_HIGH = 0.80, 1.25
ALPHA = 0.05
# ------------------------------------------


# ---------- Helpers ----------
def parse_hp_key(hp_key: str) -> Dict[str, Any]:
    """'lr=0.004, bs=4, clip=0.8' -> {'lr':0.004,'bs':4,'clip':0.8} (numeric when possible)"""
    out: Dict[str, Any] = {}
    for part in hp_key.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = [x.strip() for x in part.split("=", 1)]
        # try int -> float -> keep str
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def parse_filename_for_model_arch(path: str) -> Tuple[str, str]:
    """
    'n=2,out=6,arch=[2],model=binarized.json'
      -> ('binarized', 'n=2, out=6, arch=[2]')
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    kv = {}
    for t in stem.split(","):
        t = t.strip()
        if "=" in t:
            k, v = t.split("=", 1)
            kv[k.strip()] = v.strip()

    model_raw = (kv.get("model") or "").lower()
    if "binar" in model_raw:
        model_key = "binarized"
    elif "cont" in model_raw:
        model_key = "continuous"
    else:
        model_key = "binarized"  # default fallback

    n, outv, arch = kv.get("n"), kv.get("out"), kv.get("arch")
    arch_key = f"n={n}, out={outv}, arch={arch}" if (n and outv and arch) else stem
    return model_key, arch_key


def load_json_to_long_df(json_path: str, success_threshold: float) -> Tuple[pd.DataFrame, str, str]:
    """Return (df, model_key, arch_key). df has categoricals for seed/lr/bs/clip."""
    with open(json_path, "r") as f:
        data = json.load(f)

    model_key, arch_key = parse_filename_for_model_arch(json_path)

    lens = {k: len(v) for k, v in data.items()}
    if len(set(lens.values())) != 1:
        raise ValueError(f"Unaligned HP list lengths: {lens}")
    if not lens:
        raise ValueError("No HP entries found.")

    rows: List[Dict[str, Any]] = []
    for hp_key, acc_list in data.items():
        hp = parse_hp_key(hp_key)
        for seed_idx, acc in enumerate(acc_list):
            acc = float(acc)
            rows.append({
                "accuracy": acc,
                "success": 1 if acc >= success_threshold else 0,
                "seed": f"seed_{seed_idx}",
                "lr": hp.get("lr"),
                "bs": hp.get("bs"),
                "clip": hp.get("clip"),
            })

    df = pd.DataFrame(rows)
    # categoricals: order levels numerically if possible (baseline = first)
    df["seed"] = df["seed"].astype("category")
    for col in ["lr", "bs", "clip"]:
        lev = df[col].dropna().unique().tolist()
        try:
            lev_sorted = sorted(lev, key=float)
        except Exception:
            lev_sorted = sorted(lev, key=str)
        df[col] = pd.Categorical(df[col], categories=lev_sorted, ordered=True)

    return df, model_key, arch_key


def fit_glm(formula: str, df: pd.DataFrame):
    return smf.glm(formula, data=df, family=sm.families.Binomial()).fit()


def lrt_between(m0, m1) -> Tuple[float, int, float]:
    LR = 2.0 * (m1.llf - m0.llf)
    df = int(m1.df_model - m0.df_model)
    p = 1 - chi2.cdf(LR, df) if df > 0 else np.nan
    return LR, df, p


def additive_or_table(m) -> pd.DataFrame:
    """Main-effect ORs & 95% CIs (exclude seed and interactions)."""
    params = m.params
    conf = m.conf_int()
    rows = []
    for name, beta in params.items():
        if not name.startswith("C("):
            continue
        if name.startswith("C(seed)"):
            continue
        if ":" in name:
            continue
        try:
            var = name.split("(")[1].split(")")[0]
            level = name.split("[T.", 1)[1].rstrip("]")
        except Exception:
            var, level = name, ""
        lo, hi = conf.loc[name, 0], conf.loc[name, 1]
        rows.append({
            "term": name, "variable": var, "level_vs_baseline": level,
            "beta": beta, "OR": math.exp(beta), "OR_low": math.exp(lo), "OR_high": math.exp(hi)
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["variable", "level_vs_baseline"]).reset_index(drop=True)
    return df


def baseline_of(df: pd.DataFrame, col: str):
    cats = list(df[col].cat.categories)
    return cats[0] if cats else None


def best_lr_level_by_OR(ors_add_df: pd.DataFrame):
    lr_rows = ors_add_df[ors_add_df["variable"] == "lr"].copy()
    if lr_rows.empty:
        return None, np.nan, np.nan, np.nan
    r = lr_rows.sort_values("OR", ascending=False).iloc[0]
    return r["level_vs_baseline"], float(r["OR"]), float(r["OR_low"]), float(r["OR_high"])


def best_nonbaseline_clip_by_OR(ors_add_df: pd.DataFrame, clip_baseline) -> str:
    """Return the non-baseline clip level (as string) with highest OR; None if not available."""
    clip_rows = ors_add_df[ors_add_df["variable"] == "clip"].copy()
    if clip_rows.empty:
        return None
    base_str = to_str_level(clip_baseline)
    nonbase = clip_rows[clip_rows["level_vs_baseline"] != base_str]
    if nonbase.empty:
        return None
    r = nonbase.sort_values("OR", ascending=False).iloc[0]
    return r["level_vs_baseline"]


def to_str_level(x):
    """Unify any level (float/int/str) to a plain string for safe set/CSV use."""
    if x is None:
        return None
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        s = f"{float(x)}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    return str(x)


def within_clip_bs_tests(df: pd.DataFrame):
    """For each clip level, test bs effect while controlling for lr."""
    res = []
    for clip_val in df["clip"].cat.categories:
        d = df[df["clip"] == clip_val]
        if d.empty:
            continue
        m0 = fit_glm("success ~ C(seed) + C(lr)", d)
        m1 = fit_glm("success ~ C(seed) + C(lr) + C(bs)", d)
        LR, ddf, p = lrt_between(m0, m1)
        ors = additive_or_table(m1)
        bs_rows = ors[ors["variable"] == "bs"]
        best_bs = None
        if not bs_rows.empty:
            best_bs = bs_rows.sort_values("OR", ascending=False)["level_vs_baseline"].iloc[0]
        res.append({
            "clip": clip_val, "LR_bs": LR, "df_bs": ddf, "p_bs": p, "best_bs_by_OR": best_bs
        })
    return res


def within_clip_lr_tests(df: pd.DataFrame):
    """For each clip level, test lr effect while controlling for bs."""
    res = []
    for clip_val in df["clip"].cat.categories:
        d = df[df["clip"] == clip_val]
        if d.empty:
            continue
        m0 = fit_glm("success ~ C(seed) + C(bs)", d)
        m1 = fit_glm("success ~ C(seed) + C(bs) + C(lr)", d)
        LR, ddf, p = lrt_between(m0, m1)
        ors = additive_or_table(m1)
        lr_rows = ors[ors["variable"] == "lr"]
        best_lr = None
        if not lr_rows.empty:
            best_lr = lr_rows.sort_values("OR", ascending=False)["level_vs_baseline"].iloc[0]
        res.append({
            "clip": clip_val, "LR_lr": LR, "df_lr": ddf, "p_lr": p, "best_lr_by_OR": best_lr
        })
    return res


# -------- NEW: within-lr tests (per lr level) --------
def within_lr_bs_tests(df: pd.DataFrame):
    """For each lr level, test bs effect while controlling for clip."""
    res = []
    for lr_val in df["lr"].cat.categories:
        d = df[df["lr"] == lr_val]
        if d.empty:
            continue
        m0 = fit_glm("success ~ C(seed) + C(clip)", d)
        m1 = fit_glm("success ~ C(seed) + C(clip) + C(bs)", d)
        LR, ddf, p = lrt_between(m0, m1)
        ors = additive_or_table(m1)
        bs_rows = ors[ors["variable"] == "bs"]
        best_bs = None
        if not bs_rows.empty:
            best_bs = bs_rows.sort_values("OR", ascending=False)["level_vs_baseline"].iloc[0]
        res.append({
            "lr": lr_val, "LR_bs": LR, "df_bs": ddf, "p_bs": p, "best_bs_by_OR": best_bs
        })
    return res


def within_lr_clip_tests(df: pd.DataFrame):
    """For each lr level, test clip effect while controlling for bs."""
    res = []
    for lr_val in df["lr"].cat.categories:
        d = df[df["lr"] == lr_val]
        if d.empty:
            continue
        m0 = fit_glm("success ~ C(seed) + C(bs)", d)
        m1 = fit_glm("success ~ C(seed) + C(bs) + C(clip)", d)
        LR, ddf, p = lrt_between(m0, m1)
        ors = additive_or_table(m1)
        clip_rows = ors[ors["variable"] == "clip"]
        best_clip = None
        if not clip_rows.empty:
            best_clip = clip_rows.sort_values("OR", ascending=False)["level_vs_baseline"].iloc[0]
        res.append({
            "lr": lr_val, "LR_clip": LR, "df_clip": ddf, "p_clip": p, "best_clip_by_OR": best_clip
        })
    return res


# ---------- Per-file summary ----------
def summarize_file(path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    df, model_key, arch_key = load_json_to_long_df(path, SUCCESS_THRESHOLD)
    if model_key != "binarized":
        raise ValueError("Not a binarized file")

    # Additive base
    m_add = fit_glm("success ~ C(seed) + C(lr) + C(bs) + C(clip)", df)

    # Interactions vs additive
    m_clip_bs = fit_glm("success ~ C(seed) + C(lr) + C(clip)*C(bs)", df)
    LR_cxb, df_cxb, p_cxb = lrt_between(m_add, m_clip_bs)

    m_clip_lr = fit_glm("success ~ C(seed) + C(bs) + C(clip)*C(lr)", df)
    LR_cxl, df_cxl, p_cxl = lrt_between(m_add, m_clip_lr)

    m_bs_lr = fit_glm("success ~ C(seed) + C(clip) + C(bs)*C(lr)", df)   # NEW
    LR_bxl, df_bxl, p_bxl = lrt_between(m_add, m_bs_lr)

    # Partial LRTs (context, no interactions)
    m_seed_clip = fit_glm("success ~ C(seed) + C(clip)", df)
    m_with_lr   = fit_glm("success ~ C(seed) + C(clip) + C(lr)", df)
    m_with_bs   = fit_glm("success ~ C(seed) + C(clip) + C(bs)", df)
    LR_lr, df_lr, p_lr = lrt_between(m_seed_clip, m_with_lr)
    LR_bs, df_bs, p_bs = lrt_between(m_seed_clip, m_with_bs)

    # ORs from additive
    ors_add = additive_or_table(m_add)
    lr_best_level, lr_best_OR, lr_best_low, lr_best_high = best_lr_level_by_OR(ors_add)
    clip_base = baseline_of(df, "clip")
    clip_nonbase_best = best_nonbaseline_clip_by_OR(ors_add, clip_base)

    # Within-clip details (context)
    bs_within_clip = within_clip_bs_tests(df)
    lr_within_clip = within_clip_lr_tests(df)

    # Within-lr details (for deciding if lr can be fixed safely)
    bs_within_lr   = within_lr_bs_tests(df)
    clip_within_lr = within_lr_clip_tests(df)

    # Flags: consistency of winners across strata
    same_best_bs_across_clip = None
    winners_bs_clip = [r["best_bs_by_OR"] for r in bs_within_clip if r["best_bs_by_OR"] is not None]
    if winners_bs_clip:
        same_best_bs_across_clip = (len(set(winners_bs_clip)) == 1)

    same_best_lr_across_clip = None
    winners_lr_clip = [r["best_lr_by_OR"] for r in lr_within_clip if r["best_lr_by_OR"] is not None]
    if winners_lr_clip:
        same_best_lr_across_clip = (len(set(winners_lr_clip)) == 1)

    same_best_bs_across_lr = None
    winners_bs_lr = [r["best_bs_by_OR"] for r in bs_within_lr if r["best_bs_by_OR"] is not None]
    if winners_bs_lr:
        same_best_bs_across_lr = (len(set(winners_bs_lr)) == 1)

    same_best_clip_across_lr = None
    winners_clip_lr = [r["best_clip_by_OR"] for r in clip_within_lr if r["best_clip_by_OR"] is not None]
    if winners_clip_lr:
        same_best_clip_across_lr = (len(set(winners_clip_lr)) == 1)

    # ---- Recommendation logic ----
    # When both lr interactions are non-sig, it is safe to fix lr at the additive OR winner.
    lr_safe_to_fix = (p_cxl >= ALPHA) and (p_bxl >= ALPHA)

    rec_parts = []
    # lr recommendation
    if lr_safe_to_fix and (lr_best_level is not None):
        rec_parts.append(f"fix lr={to_str_level(lr_best_level)}")
    else:
        # build a tiny lr set suggestion
        lr_set = {to_str_level(baseline_of(df, 'lr'))}
        if lr_best_level is not None:
            lr_set.add(to_str_level(lr_best_level))
        lr_set = [x for x in lr_set if x is not None]
        if p_cxl < ALPHA or p_bxl < ALPHA:
            rec_parts.append(f"consider small lr set (due to lr interactions): lr{sorted(lr_set)}")
        else:
            rec_parts.append("consider small lr set (uncertain lr effect)")

    # clip×bs recommendation (unchanged logic)
    if p_cxb < ALPHA:
        if same_best_bs_across_clip is True:
            winner_bs = next((r["best_bs_by_OR"] for r in bs_within_clip if r["best_bs_by_OR"] is not None), None)
            rec_parts.append(f"fix bs={to_str_level(winner_bs)}, tune clip")
        elif same_best_bs_across_clip is False:
            clip_set = list({to_str_level(clip_base), to_str_level(clip_nonbase_best)} - {None})
            winners_map = {to_str_level(r["clip"]): to_str_level(r["best_bs_by_OR"]) for r in bs_within_clip}
            bs_set = list({b for b in winners_map.values() if b is not None})
            rec_parts.append(f"grid: clip{clip_set} × bs{bs_set}")
        else:
            rec_parts.append("clip×bs significant; bs winners undefined — keep small clip×bs grid")
    else:
        # no clip×bs interaction: tune whichever shows stronger main effect
        if (p_bs < ALPHA) and (p_bs <= p_lr or p_lr >= ALPHA):
            rec_parts.append("tune bs; clip additive")
        else:
            rec_parts.append("tune clip; bs additive")

    summary_row = {
        "file": os.path.basename(path),
        "arch": arch_key,
        "rows": int(df.shape[0]),
        "seeds": int(df["seed"].nunique()),
        "lr_levels": ",".join(to_str_level(x) for x in df["lr"].cat.categories.tolist()),
        "bs_levels": ",".join(to_str_level(x) for x in df["bs"].cat.categories.tolist()),
        "clip_levels": ",".join(to_str_level(x) for x in df["clip"].cat.categories.tolist()),
        # interaction tests
        "LR_clip_x_bs": LR_cxb, "df_clip_x_bs": df_cxb, "p_clip_x_bs": p_cxb,
        "LR_clip_x_lr": LR_cxl, "df_clip_x_lr": df_cxl, "p_clip_x_lr": p_cxl,
        "LR_bs_x_lr":   LR_bxl, "df_bs_x_lr":   df_bxl, "p_bs_x_lr":   p_bxl,  # NEW
        # partial LRTs
        "LR_lr": LR_lr, "df_lr": df_lr, "p_lr": p_lr,
        "LR_bs": LR_bs, "df_bs": df_bs, "p_bs": p_bs,
        # best lr by OR (additive)
        "best_lr_level": to_str_level(lr_best_level) if lr_best_level is not None else "",
        "best_lr_OR": lr_best_OR, "best_lr_OR_low": lr_best_low, "best_lr_OR_high": lr_best_high,
        # consistency flags
        "same_best_bs_across_clip": same_best_bs_across_clip,
        "same_best_lr_across_clip": same_best_lr_across_clip,
        "same_best_bs_across_lr":   same_best_bs_across_lr,     # NEW
        "same_best_clip_across_lr": same_best_clip_across_lr,   # NEW
        # safety to fix lr
        "lr_safe_to_fix": lr_safe_to_fix,                       # NEW
        # recommendation
        "recommendation": " | ".join(rec_parts)
    }

    # details per clip (type-safe string levels)
    details_clip_rows = []
    for r in bs_within_clip:
        details_clip_rows.append({
            "file": os.path.basename(path), "arch": arch_key, "stratum": f"clip={to_str_level(r['clip'])}",
            "test": "within_clip_bs", "LR": r["LR_bs"], "df": r["df_bs"], "p": r["p_bs"],
            "best_level_by_OR": to_str_level(r["best_bs_by_OR"])
        })
    for r in lr_within_clip:
        details_clip_rows.append({
            "file": os.path.basename(path), "arch": arch_key, "stratum": f"clip={to_str_level(r['clip'])}",
            "test": "within_clip_lr", "LR": r["LR_lr"], "df": r["df_lr"], "p": r["p_lr"],
            "best_level_by_OR": to_str_level(r["best_lr_by_OR"])
        })

    # details per lr (type-safe string levels) -- NEW
    details_lr_rows = []
    for r in bs_within_lr:
        details_lr_rows.append({
            "file": os.path.basename(path), "arch": arch_key, "stratum": f"lr={to_str_level(r['lr'])}",
            "test": "within_lr_bs", "LR": r["LR_bs"], "df": r["df_bs"], "p": r["p_bs"],
            "best_level_by_OR": to_str_level(r["best_bs_by_OR"])
        })
    for r in clip_within_lr:
        details_lr_rows.append({
            "file": os.path.basename(path), "arch": arch_key, "stratum": f"lr={to_str_level(r['lr'])}",
            "test": "within_lr_clip", "LR": r["LR_clip"], "df": r["df_clip"], "p": r["p_clip"],
            "best_level_by_OR": to_str_level(r["best_clip_by_OR"])
        })

    return summary_row, details_clip_rows, details_lr_rows


# --------------- Main ---------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    details_clip_all: List[Dict[str, Any]] = []
    details_lr_all: List[Dict[str, Any]] = []

    files = [os.path.join(RESULT_DIR, f) for f in os.listdir(RESULT_DIR) if f.endswith(".json")]
    files.sort()

    for path in files:
        try:
            model_key, _ = parse_filename_for_model_arch(path)
            if model_key != "binarized":
                continue
            summary_row, details_clip_rows, details_lr_rows = summarize_file(path)
            summary_rows.append(summary_row)
            details_clip_all.extend(details_clip_rows)
            details_lr_all.extend(details_lr_rows)
        except Exception as e:
            # Keep an error row so you can see which file failed and why
            summary_rows.append({
                "file": os.path.basename(path),
                "arch": "",
                "rows": np.nan,
                "seeds": np.nan,
                "lr_levels": "",
                "bs_levels": "",
                "clip_levels": "",
                "LR_clip_x_bs": np.nan, "df_clip_x_bs": np.nan, "p_clip_x_bs": np.nan,
                "LR_clip_x_lr": np.nan, "df_clip_x_lr": np.nan, "p_clip_x_lr": np.nan,
                "LR_bs_x_lr":   np.nan, "df_bs_x_lr":   np.nan, "p_bs_x_lr":   np.nan,
                "LR_lr": np.nan, "df_lr": np.nan, "p_lr": np.nan,
                "LR_bs": np.nan, "df_bs": np.nan, "p_bs": np.nan,
                "best_lr_level": "",
                "best_lr_OR": np.nan, "best_lr_OR_low": np.nan, "best_lr_OR_high": np.nan,
                "same_best_bs_across_clip": np.nan,
                "same_best_lr_across_clip": np.nan,
                "same_best_bs_across_lr":   np.nan,
                "same_best_clip_across_lr": np.nan,
                "lr_safe_to_fix": np.nan,
                "recommendation": f"ERROR: {e}",
            })

    # Write CSVs
    summary_df = pd.DataFrame(summary_rows).sort_values(["arch", "file"])
    details_clip_df = pd.DataFrame(details_clip_all).sort_values(["arch", "file", "stratum", "test"])
    details_lr_df   = pd.DataFrame(details_lr_all).sort_values(["arch", "file", "stratum", "test"])

    os.makedirs(OUT_DIR, exist_ok=True)
    sum_path  = os.path.join(OUT_DIR, "binarized_interactions_summary.csv")
    detc_path = os.path.join(OUT_DIR, "binarized_within_clip_details.csv")
    detl_path = os.path.join(OUT_DIR, "binarized_within_lr_details.csv")

    summary_df.to_csv(sum_path, index=False)
    details_clip_df.to_csv(detc_path, index=False)
    details_lr_df.to_csv(detl_path, index=False)

    print(f"[OK] Wrote {sum_path} ({len(summary_df)} rows)")
    print(f"[OK] Wrote {detc_path} ({len(details_clip_df)} rows)")
    print(f"[OK] Wrote {detl_path} ({len(details_lr_df)} rows)")


if __name__ == "__main__":
    main()

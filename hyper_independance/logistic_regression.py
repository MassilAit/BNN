# lrt_folder_summary.py
# ------------------------------------------------------------
# Folder-wide analysis:
#   - For each JSON in Result/: run Global LRT + Partial LRTs + ORs (95% CI)
#   - Continuous: ignore 'clip' (generation bug)
#   - Outputs (CSV):
#       out/lrt_summary_binarized.csv
#       out/lrt_summary_continuous.csv
#       out/ors_binarized.csv
#       out/ors_continuous.csv
# ------------------------------------------------------------

import os
import json
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

# -------- Settings --------
RESULT_DIR = "Result"
OUT_DIR = "out"
SUCCESS_THRESHOLD = 100.0   # accuracy >= threshold => success=1
# --------------------------


# -------- Helpers --------
def parse_hp_key(hp_key: str) -> Dict[str, Any]:
    """'lr=0.004, bs=4, clip=0.8' -> {'lr':0.004, 'bs':4, 'clip':0.8}"""
    out: Dict[str, Any] = {}
    for part in hp_key.split(","):
        part = part.strip()
        if "=" not in part:
            continue
        k, v = [x.strip() for x in part.split("=", 1)]
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
      -> model_key ('binarized'|'continuous'), arch_label 'n=..., out=..., arch=[...]'
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    tokens = [t.strip() for t in stem.split(",")]
    kv = {}
    for t in tokens:
        if "=" in t:
            k, v = t.split("=", 1)
            kv[k.strip()] = v.strip()

    model_raw = (kv.get("model") or "").lower()
    if "binar" in model_raw:
        model_key = "binarized"
    elif "cont" in model_raw:
        model_key = "continuous"
    else:
        # default if missing; most of your files encode it
        model_key = "binarized"

    n = kv.get("n"); out = kv.get("out"); arch = kv.get("arch")
    if n is not None and out is not None and arch is not None:
        arch_key = f"n={n}, out={out}, arch={arch}"
    else:
        arch_tokens = [t for t in tokens if not t.startswith("model=")]
        arch_key = ", ".join(arch_tokens) if arch_tokens else stem
    return model_key, arch_key


def load_json_to_long_df(json_path: str, success_threshold: float) -> Tuple[pd.DataFrame, str, str]:
    """
    Returns (df, model_key, arch_key).
    df columns: ['success','accuracy','seed','lr','bs','clip'] (categorical)
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    model_key, arch_key = parse_filename_for_model_arch(json_path)

    # Validate aligned lengths
    lengths = {hp: len(accs) for hp, accs in data.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Unaligned HP list lengths: {lengths}")
    n_seeds = next(iter(lengths.values()))
    if n_seeds == 0:
        raise ValueError("Zero-length accuracy lists.")

    rows: List[Dict[str, Any]] = []
    for hp_key, acc_list in data.items():
        hp = parse_hp_key(hp_key)
        for seed_idx, acc in enumerate(acc_list):
            rows.append({
                "accuracy": float(acc),
                "success": 1 if float(acc) >= success_threshold else 0,
                "seed": f"seed_{seed_idx}",
                "lr": hp.get("lr"),
                "bs": hp.get("bs"),
                "clip": hp.get("clip"),
            })

    df = pd.DataFrame(rows)

    # Ensure categorical types (ordered by sorted levels; baseline = first)
    df["seed"] = df["seed"].astype("category")
    for col in ["lr", "bs", "clip"]:
        levels = sorted(df[col].dropna().unique().tolist())
        df[col] = pd.Categorical(df[col], categories=levels, ordered=True)

    return df, model_key, arch_key


def build_formulas(model_key: str) -> Tuple[str, str, List[str]]:
    """
    Returns (null_formula, full_formula, hp_vars).
    Continuous: hp_vars = ['lr','bs']
    Binarized : hp_vars = ['lr','bs','clip']
    """
    null_f = "success ~ C(seed)"
    hp_vars = ["lr", "bs"] if model_key == "continuous" else ["lr", "bs", "clip"]
    full_terms = " + ".join([f"C({v})" for v in hp_vars])
    full_f = f"{null_f} + {full_terms}" if full_terms else null_f
    return null_f, full_f, hp_vars


def fit_glm(formula: str, df: pd.DataFrame):
    return smf.glm(formula, data=df, family=sm.families.Binomial()).fit()


def lrt_between(m0, m1) -> Dict[str, float]:
    ll0, ll1 = m0.llf, m1.llf
    df_diff = int(m1.df_model - m0.df_model)
    LR = 2.0 * (ll1 - ll0)
    pval = 1.0 - chi2.cdf(LR, df_diff) if df_diff > 0 else np.nan
    return {"LL0": ll0, "LL1": ll1, "df": df_diff, "LR": LR, "p": pval}


def run_global_and_partial_lrts(df: pd.DataFrame, model_key: str):
    null_f, full_f, hp_vars = build_formulas(model_key)
    m0 = fit_glm(null_f, df)
    m_full = fit_glm(full_f, df)
    global_res = lrt_between(m0, m_full)

    partial = {}
    for var in hp_vars:
        m_alt = fit_glm(f"success ~ C(seed) + C({var})", df)
        partial[var] = lrt_between(m0, m_alt)
    return m0, m_full, global_res, partial, hp_vars


def tidy_or_table(m_full, file_name: str, model_key: str, arch_key: str) -> pd.DataFrame:
    """
    Extract OR & 95% CI for all HP levels (excluding seed dummies).
    Returns long DF with columns:
      [file, model, arch, variable, level_vs_baseline, beta, OR, OR_CI_low, OR_CI_high]
    """
    params = m_full.params
    conf = m_full.conf_int()
    rows = []
    for name, beta in params.items():
        if not name.startswith("C(") or name.startswith("C(seed)"):
            continue
        try:
            var = name.split("(")[1].split(")")[0]
            level = name.split("[T.", 1)[1].rstrip("]")
        except Exception:
            var = name
            level = ""
        lo, hi = conf.loc[name, 0], conf.loc[name, 1]
        rows.append({
            "file": file_name,
            "model": model_key,
            "arch": arch_key,
            "variable": var,
            "level_vs_baseline": level,
            "beta": beta,
            "OR": np.exp(beta),
            "OR_CI_low": np.exp(lo),
            "OR_CI_high": np.exp(hi),
        })
    return pd.DataFrame(rows)


# --------------- Main ---------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    or_rows: List[pd.DataFrame] = []

    files = [os.path.join(RESULT_DIR, f) for f in os.listdir(RESULT_DIR) if f.endswith(".json")]
    files.sort()

    for path in files:
        fname = os.path.basename(path)
        try:
            df, model_key, arch_key = load_json_to_long_df(path, SUCCESS_THRESHOLD)

            # Fit models & tests
            m0, m_full, global_res, partial, hp_vars = run_global_and_partial_lrts(df, model_key)

            # Summaries
            row = {
                "file": fname,
                "model": model_key,
                "arch": arch_key,
                "rows": int(df.shape[0]),
                "seeds": int(df["seed"].nunique()),
                "levels_lr": ",".join(map(str, df["lr"].cat.categories.tolist())) if "lr" in df else "",
                "levels_bs": ",".join(map(str, df["bs"].cat.categories.tolist())) if "bs" in df else "",
                "levels_clip": ",".join(map(str, df["clip"].cat.categories.tolist())) if "clip" in df and model_key=="binarized" else "",
                # Global LRT
                "LL_null": global_res["LL0"],
                "LL_full": global_res["LL1"],
                "df_global": global_res["df"],
                "LR_global": global_res["LR"],
                "p_global": global_res["p"],
                "AIC_null": m0.aic,
                "AIC_full": m_full.aic,
                "dAIC": m_full.aic - m0.aic,
            }
            # Partial LRTs (fill NaN if var not in hp_vars)
            for v in ["lr", "bs", "clip"]:
                if v in hp_vars:
                    row[f"df_{v}"] = partial[v]["df"]
                    row[f"LR_{v}"] = partial[v]["LR"]
                    row[f"p_{v}"]  = partial[v]["p"]
                else:
                    row[f"df_{v}"] = np.nan
                    row[f"LR_{v}"] = np.nan
                    row[f"p_{v}"]  = np.nan

            summary_rows.append(row)

            # OR table
            or_rows.append(tidy_or_table(m_full, fname, model_key, arch_key))

        except Exception as e:
            # Record error row so you can see which file failed
            summary_rows.append({
                "file": fname,
                "model": "unknown",
                "arch": "unknown",
                "rows": np.nan,
                "seeds": np.nan,
                "levels_lr": "",
                "levels_bs": "",
                "levels_clip": "",
                "LL_null": np.nan,
                "LL_full": np.nan,
                "df_global": np.nan,
                "LR_global": np.nan,
                "p_global": np.nan,
                "AIC_null": np.nan,
                "AIC_full": np.nan,
                "dAIC": np.nan,
                "df_lr": np.nan, "LR_lr": np.nan, "p_lr": np.nan,
                "df_bs": np.nan, "LR_bs": np.nan, "p_bs": np.nan,
                "df_clip": np.nan, "LR_clip": np.nan, "p_clip": np.nan,
                "error": str(e),
            })
            print(f"[WARN] Skipped {fname}: {e}")

    # Build DataFrames
    summary_df = pd.DataFrame(summary_rows)
    ors_df = pd.concat(or_rows, ignore_index=True) if len(or_rows) else pd.DataFrame(
        columns=["file","model","arch","variable","level_vs_baseline","beta","OR","OR_CI_low","OR_CI_high"]
    )

    # Split by model and save
    for model_type in ["continuous", "binarized"]:
        summ = summary_df[summary_df["model"] == model_type].sort_values(["arch","file"])
        ors  = ors_df[ors_df["model"] == model_type].sort_values(["arch","file","variable","level_vs_baseline"])

        summ_path = os.path.join(OUT_DIR, f"lrt_summary_{model_type}.csv")
        ors_path  = os.path.join(OUT_DIR, f"ors_{model_type}.csv")
        summ.to_csv(summ_path, index=False)
        ors.to_csv(ors_path, index=False)
        print(f"[OK] Wrote {summ_path}  ({len(summ)} rows)")
        print(f"[OK] Wrote {ors_path}   ({len(ors)} rows)")


if __name__ == "__main__":
    main()

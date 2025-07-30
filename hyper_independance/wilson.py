# wilson_subplots.py
import os
import json
import math
from math import sqrt
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULT_DIR = "Result"        # change if needed
SUCCESS_THRESHOLD = 100.0    # accuracy threshold counted as a success
Z_95 = 1.96                  # 95% CI

# ---------------- Utils ----------------
def wilson(successes: int, n: int, z: float = Z_95):
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + (z**2)/n
    center = p + (z**2)/(2*n)
    margin = z * sqrt((p*(1-p) + (z**2)/(4*n)) / n)
    return (center - margin) / denom, (center + margin) / denom

def parse_filename(stem: str):
    """
    Parse 'n=2,out=6,arch=[2],model=binarized' -> (arch_key, model_key)
    """
    parts = [t.strip() for t in stem.split(",")]
    d: Dict[str, Any] = {}
    for t in parts:
        if "=" in t:
            k, v = t.split("=", 1)
            d[k.strip()] = v.strip()
    n = d.get("n"); out = d.get("out"); arch = d.get("arch")
    if n is not None and out is not None and arch is not None:
        arch_key = f"n={n}, out={out}, arch={arch}"
    else:
        arch_key = ", ".join([t for t in parts if not t.startswith("model=")]) or stem

    model_raw = (d.get("model") or "").lower()
    model_key = "binarized" if "binar" in model_raw else "continuous"
    return arch_key, model_key

def collect_rows(result_dir: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fname in os.listdir(result_dir):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(result_dir, fname)
        arch_key, model_key = parse_filename(os.path.splitext(fname)[0])
        with open(path, "r") as f:
            data = json.load(f)
        for hp_key, accs in data.items():
            accs = [float(a) for a in accs]
            n = len(accs)
            s = sum(1 for a in accs if a >= SUCCESS_THRESHOLD)
            low, high = wilson(s, n, Z_95)
            p_hat = s / n if n else 0.0
            rows.append({
                "file": fname, "model": model_key, "arch": arch_key, "hp": hp_key,
                "n": n, "successes": s, "p_hat": p_hat,
                "wilson_low": low, "wilson_high": high
            })
    return pd.DataFrame(rows).sort_values(["model", "arch", "file", "hp"]).reset_index(drop=True)

# ------------- Plot: 1 figure with subplots (per model) -------------
def plot_model_subplots(df: pd.DataFrame, model_key: str, out_path: str,
                        jitter=0.05, markersize=3, elinewidth=0.8, capsize=2):
    dfm = df[df["model"] == model_key]
    arches = sorted(dfm["arch"].unique())
    if len(arches) == 0:
        print(f"[warn] no data for model={model_key}")
        return

    n_arch = len(arches)          # e.g., 7
    cols = min(4, n_arch)         # layout: up to 4 columns
    rows = math.ceil(n_arch / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 4.8*rows), sharey=True)
    axes = np.array(axes).reshape(-1)  # flatten even if rows*cols==1

    for i, arch in enumerate(arches):
        ax = axes[i]
        dfa = dfm[dfm["arch"] == arch].reset_index(drop=True)
        k = len(dfa)
        xs = np.linspace(-jitter, jitter, k) if k > 1 else np.array([0.0])
        ys = dfa["p_hat"].values
        yerr = np.vstack([ys - dfa["wilson_low"].values,
                          dfa["wilson_high"].values - ys])

        ax.errorbar(xs, ys, yerr=yerr, fmt="o", capsize=capsize,
                    markersize=markersize, elinewidth=elinewidth, alpha=0.9)
        ax.set_title(f"{arch}\n({k} HP combos)", fontsize=10)
        ax.set_xticks([])
        ax.set_ylim(0.0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Hide any unused axes (if rows*cols > n_arch)
    for j in range(len(arches), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Wilson intervals (95%) — model={model_key}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

# -------------------- Run --------------------
if __name__ == "__main__":
    df = collect_rows(RESULT_DIR)
    # One figure per model, each with 7 (or N) subplots:
    plot_model_subplots(df, "binarized",  "wilson_binarized_subplots.png")
    plot_model_subplots(df, "continuous", "wilson_continuous_subplots.png")

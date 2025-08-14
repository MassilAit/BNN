# save as compare_min_arch.py
import pandas as pd
import numpy as np
import re

# ---- hardcode your file paths here ----
n=4
FPATH_458 = f"{n}_continious_output.csv"
FPATH_10  = f"{n}_continuous_output_small.csv"
ONLY_SUCCESS = True  # set False if you want to include all rows

def extract_first_int(text):
    if pd.isna(text):
        return np.nan
    nums = re.findall(r"-?\d+", str(text))
    return int(nums[0]) if nums else np.nan

def load_min_per_canonical(path, label):
    df = pd.read_csv(path)
    if ONLY_SUCCESS and "success" in df.columns:
        df = df[df["success"] == 1]

    df = df.copy()
    df["canonical"] = pd.to_numeric(df["canonical"], errors="coerce")
    df["arch_num"] = df["min_architecture"].apply(extract_first_int)

    df = df.dropna(subset=["canonical", "arch_num"])
    g = df.groupby("canonical", as_index=False)["arch_num"].min()
    g = g.rename(columns={"arch_num": f"arch_{label}"})
    return g

a = load_min_per_canonical(FPATH_458, "458")
b = load_min_per_canonical(FPATH_10,  "10")

merged = pd.merge(a, b, on="canonical", how="inner")
if merged.empty:
    print("No overlapping canonicals between the two files after filtering.")
    raise SystemExit

merged["delta"] = merged["arch_458"] - merged["arch_10"]

gt = (merged["arch_458"] > merged["arch_10"]).sum()
lt = (merged["arch_458"] < merged["arch_10"]).sum()
eq = (merged["arch_458"] == merged["arch_10"]).sum()
median_delta = merged["delta"].median()

print(f"========== {n} inputs ========")

print(f"Compared canonicals (intersection): {len(merged)}")
print(f"> 458-retry arch > 10-retry arch: {gt}")
print(f"< 458-retry arch < 10-retry arch: {lt}")
print(f"= 458-retry arch = 10-retry arch: {eq}")
print(f"Median difference (arch_458 - arch_10): {median_delta:.4f}")

# Biggest absolute difference
idx_abs = merged["delta"].abs().idxmax()
row_abs = merged.loc[idx_abs]
direction = "higher" if row_abs["delta"] > 0 else ("lower" if row_abs["delta"] < 0 else "equal")

print(f"\nLargest absolute difference:delta={int(row_abs['delta'])} → 458-retry is {direction})")

# (optional) uncomment to see which canonicals exist only in one file
# only_in_458 = set(a["canonical"]) - set(b["canonical"])
# only_in_10   = set(b["canonical"]) - set(a["canonical"])
# print(f"\nOnly in 458 file: {len(only_in_458)}; Only in 10 file: {len(only_in_10)}")

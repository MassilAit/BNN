# save as analyze_arch.py
import pandas as pd
import numpy as np
import re


n=4
FPATH = f"Result/{n}_continuous.csv"   # <-- set your CSV path


def extract_first_int(text):
    if pd.isna(text):
        return np.nan
    nums = re.findall(r"-?\d+", str(text))
    return int(nums[0]) if nums else np.nan

df = pd.read_csv(FPATH)

# Parse numeric values
df["min_architecture_num"] = df["min_architecture"].apply(extract_first_int)  # neurons
df["sop_pos_terms"] = pd.to_numeric(df["min_arch"], errors="coerce")          # SOP/POS terms
d = df.dropna(subset=["min_architecture_num", "sop_pos_terms"]).copy()
d["delta"] = d["min_architecture_num"] - d["sop_pos_terms"]

if d.empty:
    print("No valid rows after parsing 'min_architecture' and 'min_arch'.")
    raise SystemExit

# Counts
gt = (d["min_architecture_num"] > d["sop_pos_terms"]).sum()
lt = (d["min_architecture_num"] < d["sop_pos_terms"]).sum()
eq = (d["min_architecture_num"] == d["sop_pos_terms"]).sum()

# Median delta
median_delta = d["delta"].median()

print(f"========= {n}-intputs ==========")

print(f"> min_architecture (neurons) >  SOP/POS terms: {gt}")
print(f"< min_architecture (neurons) <  SOP/POS terms: {lt}")
print(f"= min_architecture (neurons) == SOP/POS terms: {eq}")
print(f"Median delta (min_architecture - SOP/POS terms): {median_delta:.4f}")

# Biggest differences
idx_max = d["delta"].idxmax()
idx_min = d["delta"].idxmin()
idx_abs = d["delta"].abs().idxmax()

def row_summary(row):
    can = row.get("canonical", "")
    n_in = row.get("n_input", "")
    return (
        f"(n_input={n_in}, canonical={can}, "
        f"min_architecture={int(row['min_architecture_num'])}, "
        f"SOP/POS terms={int(row['sop_pos_terms'])}, "
        f"delta={int(row['delta'])})"
    )

row_max = d.loc[idx_max]
row_min = d.loc[idx_min]
row_abs = d.loc[idx_abs]

print("Max negative delta (min_architecture lower):", int(row_min['delta']))


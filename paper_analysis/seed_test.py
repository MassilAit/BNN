import pandas as pd

FILE = "4_binarized_output.csv"  # <-- put your filename here

TARGETS = [278, 280, 300, 318, 360, 362, 382, 391, 408, 494, 510, 829, 854, 874, 893, 894, 1973]

df = pd.read_csv(FILE)

# Coerce types just in case they’re strings
df["canonical"] = pd.to_numeric(df["canonical"], errors="coerce")
df["seed"] = pd.to_numeric(df["seed"], errors="coerce")

# Drop rows without a valid seed
df = df.dropna(subset=["seed"])

overall_mean = df["seed"].mean()

mask = df["canonical"].isin(TARGETS)
filtered = df[mask]
filtered_mean = filtered["seed"].mean() if not filtered.empty else float("nan")

print(f"Overall average seed: {overall_mean:.3f} (n={len(df)})")
print(f"Average seed for TARGET canonicals: {filtered_mean:.3f} (n={len(filtered)})")

# Optional: sanity checks
missing = sorted(set(TARGETS) - set(df.loc[mask, "canonical"].dropna().astype(int)))
if missing:
    print("Canonicals in TARGETS not found in the file:", missing)

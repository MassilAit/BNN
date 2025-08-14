import pandas as pd, ast, re

CONT = "Result/4_continuous.csv"
BIN  = "Result/4_binarized.csv"

def total_neurons(val):
    if pd.isna(val): return None
    s = str(val).strip()
    try:
        obj = ast.literal_eval(s)
    except Exception:
        s2 = s.replace('"','').replace("'",'')
        try:
            obj = ast.literal_eval(s2)
        except Exception:
            nums = re.findall(r'-?\d+', s)
            obj = [int(x) for x in nums] if nums else None
    if obj is None: return None
    if isinstance(obj, int): return int(obj)
    if isinstance(obj, (list, tuple)): return int(sum(int(x) for x in obj))
    return None

def load(path):
    df = pd.read_csv(path)
    # prefer min_architecture; fallback to min_arch if needed
    if "min_architecture" in df.columns:
        neurons = df["min_architecture"].apply(total_neurons)
    else:
        neurons = df["min_arch"].apply(lambda x: int(str(x)) if pd.notna(x) else None)
    return df.assign(total_hidden_neurons=neurons)[["n_input","canonical","total_hidden_neurons"]]

cont = load(CONT).rename(columns={"total_hidden_neurons":"cont_neurons"})
binf = load(BIN).rename(columns={"total_hidden_neurons":"bin_neurons"})

m = pd.merge(cont, binf, on=["n_input","canonical"], how="inner")

bin_better  = m[m["bin_neurons"]  < m["cont_neurons"]].copy()
cont_better = m[m["cont_neurons"] < m["bin_neurons"]].copy()
ties        = m[m["cont_neurons"] == m["bin_neurons"]].copy()

print("Binarized uses fewer neurons (canonical list):")
print(sorted(bin_better["canonical"].tolist()))
print(f"count = {len(bin_better)}")

print("\nContinuous uses fewer neurons (canonical list):")
print(sorted(cont_better["canonical"].tolist()))
print(f"count = {len(cont_better)}")

print("\nTies (same neuron count):")
print(sorted(ties["canonical"].tolist()))
print(f"count = {len(ties)}")

# optional: save detail tables
bin_better.to_csv("canonicals_binarized_better.csv", index=False)
cont_better.to_csv("canonicals_continuous_better.csv", index=False)
ties.to_csv("canonicals_ties.csv", index=False)

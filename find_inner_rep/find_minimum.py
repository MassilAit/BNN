# COMPLETE SCRIPT — set N_INPUT and file paths below.
# Reads a CSV of successful runs, retrains each model with the row's HP+seed,
# extracts inner representations, and writes them with:
#   - SOP/POS preferred expression (from JSON)
#   - Network stats (two_input_eq(tree,dag), depth_no_not, gate_count per gate (tree,dag))
#   - SOP/POS stats computed the SAME way (by passing the SOP/POS expression string)
#   - Best implementation (network vs SOP/POS) using multilevel_cost (DAG-preferred)
#   - min_arch = preferred SOP/POS terms count (first element of preferred (terms, literals))

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import pandas as pd
import ast
import json
import csv
from pathlib import Path

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from find_logic import minimise_one_ones
from model   import ContiniousdModel, BinarizedModel
from dataset import LogicDataSet
from train   import train_model

# ─────────────────────────────────────────────────────────────────────────────
#  USER CONFIG
# ─────────────────────────────────────────────────────────────────────────────
N_INPUT: int = 3  # <<< SET THIS
MODEL_TYPE : str =  "binarized" #"continuous" 
INPUT_CSV      = f"{N_INPUT}_{MODEL_TYPE}_output.csv"
BEST_FORM_JSON = "npn_classes_bool.json"
OUTPUT_CSV     = f"Result/{N_INPUT}_{MODEL_TYPE}.csv"

# ─────────────────────────────────────────────────────────────────────────────
#  Metric glue (uniform stats for any expression string)
# ─────────────────────────────────────────────────────────────────────────────
from metric import get_expression_stats, _multilevel_cost  # YOUR function: analyze_multilevel(_parse(expr))


def _gate_counts_tuple(gc_tree: dict, gc_dag: dict) -> dict:
    """
    Convert raw gate_counts dicts into:
      {"and": (tree_and, dag_and), "or": (tree_or, dag_or), "not": (tree_not, dag_not)}
    """
    return {
        "and": (int(gc_tree.get("and", 0)), int(gc_dag.get("and", 0))),
        "or":  (int(gc_tree.get("or",  0)), int(gc_dag.get("or",  0))),
        "not": (int(gc_tree.get("not", 0)), int(gc_dag.get("not", 0))),
    }

def summarize_expr(expr: str) -> dict:
    """
    Return simplified stats for any expression string:
      - cost: (tree, dag)
      - depth_no_not: int
      - gate_count: {"and": (tree_and, dag_and), "or": (tree_or, dag_or), "not": (tree_not, dag_not)}
      - cost_tuple: DAG-preferred multilevel cost (for comparison)
    """
    try:
        raw = get_expression_stats(expr)
        cost_tree = int(raw["tree"]["cost"])
        cost_dag  = int(raw["dag"]["cost"])
        depth    = int(raw["tree"]["depth_no_not"])
        gate_ct  = _gate_counts_tuple(raw["tree"]["gate_counts"], raw["dag"]["gate_counts"])
        cost_key = _multilevel_cost(raw, prefer="dag")
        return {
            "cost": (cost_tree, cost_dag),
            "depth_no_not": depth,
            "gate_count": gate_ct,
            "cost_tuple": cost_key,
        }
    except Exception as e:
        # make it sortable; huge sentinels
        return {
            "cost": (10**9, 10**9),
            "depth_no_not": 10**9,
            "gate_count": {"and": (10**9, 10**9), "or": (10**9, 10**9), "not": (10**9, 10**9)},
            "cost_tuple": (10**9, 10**9, 10**9),
            "error": f"{type(e).__name__}: {e}",
        }

# ─────────────────────────────────────────────────────────────────────────────
#  Hyper-parameter container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Hyper:
    """Bundle of all tunable parameters."""
    lr:            float  = 0.002       # Adam learning rate
    delta:         float  = 0.01        # Early-stopping min-Δ
    patience:      int    = 100         # Early-stopping patience
    attempts:      int    = 50          # Re-trainings per architecture (unused here; single run)
    epochs:        int    = 10_000      # Max epochs per training
    batch_size:    Optional[int] = None # If None → (2**n_input)//4
    model_type:    str    = "binarized" # 'continuous' | 'binarized'
    clip:          float  = 0.8         # clipping for STE (binarized only)

# ─────────────────────────────────────────────────────────────────────────────
#  Seed & DataLoader helpers
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def make_loader(dataset, batch_size, seed):
    g = torch.Generator()
    g.manual_seed(seed)  # fixes the shuffle order across runs
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)

# ─────────────────────────────────────────────────────────────────────────────
#  Loss & evaluation
# ─────────────────────────────────────────────────────────────────────────────
class BCEPlusMinusOne(nn.Module):
    """BCEWithLogitsLoss adapted for {-1, +1} targets."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets_pm1):
        targets_01 = (targets_pm1 + 1) / 2  # −1→0, +1→1
        return self.bce(logits, targets_01)

def evaluate_model_accuracy(dataset: LogicDataSet, model: nn.Module) -> float:
    """
    Compute percentage accuracy over a LogicDataSet.
    • Model is expected to output a logit for each sample.
    • Prediction sign = 0 is interpreted as +1 by convention.
    """
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y_true in dataset:
            logit = model(X.unsqueeze(0))
            y_pred = torch.sign(logit)[0][0]
            y_true = y_true[0] if y_true.dim() else y_true
            if y_pred.item() == 0:
                y_pred = torch.tensor(1., dtype=y_true.dtype)
            correct += (y_pred.item() == y_true.item())
            total   += 1
    return 100.0 * correct / total

# ─────────────────────────────────────────────────────────────────────────────
#  Training wrapper
# ─────────────────────────────────────────────────────────────────────────────
EPS = 1e-9  # near-100% tolerance

def try_training(
    n_input: int,
    hidden_layers: List[int],
    train_ds: LogicDataSet,
    hp: Hyper,
    seed : int
) -> Tuple[float, nn.Module]:
    """Train one model and return (accuracy_on_train, trained_model)."""
    set_seed(seed)

    if hp.model_type.lower() == "continuous":
        model = ContiniousdModel(n_input, hidden_layers, bias=True)
    else:
        model = BinarizedModel(n_input, hidden_layers, bias=True, alpha=hp.clip)

    batch = hp.batch_size if hp.batch_size is not None else max(1, (2**n_input)//4)
    loader = make_loader(train_ds, batch, seed)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = BCEPlusMinusOne()

    train_model(
        model, loader, optimizer,
        hp.epochs, criterion,
        early_stop=True, delta=hp.delta, patience=hp.patience
    )

    return evaluate_model_accuracy(train_ds, model), model

def substitute_hidden(expr: str, hidden_exprs: List[str]) -> str:
    """
    Replace A, B, C, … in expr with their corresponding hidden expressions
    inside [ ] brackets, even if they appear concatenated like 'AB'.
    """
    out = ""
    for char in expr:
        if 'A' <= char <= 'Z':
            idx = ord(char) - ord('A')
            if idx < len(hidden_exprs):
                out += f"[{hidden_exprs[idx]}]"
            else:
                out += char
        else:
            out += char
    return out

def binary_list_to_int(bits: List[int]) -> int:
    """
    bits: list[int] of 0/1
    returns: integer after reversing the list (LSB = first hidden)
    e.g., [1,0] → [0,1] → 1
    """
    bits = bits[::-1]
    value = 0
    for i, b in enumerate(bits):
        value |= (b << i)
    return value

# ───────────────────────────────────────── analyse whole network ──
def analyse_model(model: nn.Module, n_inputs: int, eval_ds: LogicDataSet, n_hidden: List[int]) -> Any:
    # No hidden layer → directly minimize the output minterms
    if len(n_hidden) == 0:
        minterms = []
        with torch.no_grad():
            for i, (X, _) in enumerate(eval_ds):
                y_pred = torch.sign(model(X.unsqueeze(0))).item()
                if y_pred >= 0:
                    minterms.append(i)
        expr = minimise_one_ones(minterms, n_inputs)
        return expr  # direct expression

    # Single hidden layer (current support)
    hidden_minterms = [[] for _ in range(n_hidden[0])]
    output_true: set[int]  = set()
    output_false: set[int] = set()
    full_set = set(range(2**n_hidden[0]))

    with torch.no_grad():
        for i, (X, _) in enumerate(eval_ds):
            y_pred = torch.sign(model(X.unsqueeze(0))).item()

            # hidden layer activations
            activ = model.get_activations()["act_0"][0].tolist()
            activ = [1 if a >= 0 else 0 for a in activ]  # map -1/+1 → 0/1

            # record minterms for hidden neurons
            for h, h_val in enumerate(activ):
                if h_val == 1:
                    hidden_minterms[h].append(i)

            if y_pred == 1.0:
                output_true.add(binary_list_to_int(activ))
            else:
                output_false.add(binary_list_to_int(activ))

    dont_cares = full_set - (output_true | output_false)
    output_expression = minimise_one_ones(list(output_true), n_hidden[0], list(dont_cares))
    hidden_exprs = [minimise_one_ones(m, n_inputs) for m in hidden_minterms]
    full_expr = substitute_hidden(output_expression, hidden_exprs)
    return hidden_exprs, output_expression, full_expr

def find_inner(
    n_input: int,
    canonical: int,
    arch: List[int],
    seed : int,
    hp: Hyper,
) -> Any:
    dataset = LogicDataSet(n_input, canonical)
    acc, model = try_training(n_input, arch, dataset, hp, seed)

    if acc == 100.0:
        return analyse_model(model, n_input, dataset, arch)

    print(f"Error for {canonical}")
    return "","",""

# ─────────────────────────────────────────────────────────────────────────────
#  JSON helpers for SOP/POS string + min_arch (preferred term count)
# ─────────────────────────────────────────────────────────────────────────────
def _load_preferred_terms(best_json_path: str, n_input: int, canonical: int) -> str:
    """
    Return the preferred terms string (SOP_terms or POS_terms) for the
    Boolean function (n_input, canonical). If not found → "".
    """
    with open(best_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nk, ck = str(n_input), str(canonical)
    if nk in data and ck in data[nk]:
        entry = data[nk][ck]
        return str(entry.get("POS_terms" if entry.get("preferred", "SOP") == "POS" else "SOP_terms", ""))
    return ""

def _load_preferred_terms_count(best_json_path: str, n_input: int, canonical: int) -> Optional[int]:
    """
    Return the first element (#terms) of preferred cost (SOP_cost or POS_cost).
    If not found, return None.
    """
    with open(best_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nk, ck = str(n_input), str(canonical)
    entry = data.get(nk, {},).get(ck, {})
    preferred = entry.get("preferred", "SOP")
    cost = entry.get("POS_cost" if preferred == "POS" else "SOP_cost", None)
    if isinstance(cost, list) or isinstance(cost, tuple):
        return int(cost[0]) if len(cost) >= 1 else None
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def process_rows(input_csv: str, best_json: str, output_csv: str, n_input: int) -> None:
    df = pd.read_csv(input_csv)
    if "success" in df.columns:
        df = df[df["success"] == True].copy()

    out_fields = [
        "n_input", "canonical", "min_architecture",     # existing architecture column
        "hidden_expressions", "output_expression",
        "inner_representation", "SOP/POS",
        # NEW (only the fields you asked for):
        "network_stats", "sop_pos_stats", "best_impl", "min_arch"
    ]

    Path(Path(output_csv).parent).mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()

        for _, row in df.iterrows():
            try:
                canonical = int(row["canonical"])
                arch      = _parse_arch(row.get("min_architecture", []))
                seed_val  = int(row.get("seed", 0))
                hp        = _hp_from_row(row)

                result = find_inner(n_input, canonical, arch, seed_val, hp)

                # normalize result shape
                if isinstance(result, tuple):
                    hidden_exprs, output_expr, full_expr = result
                else:  # no hidden layer or training error
                    hidden_exprs, output_expr, full_expr = [], str(result), str(result)

                # SOP/POS expression string from JSON (preferred)
                sop_pos_expr = _load_preferred_terms(best_json, n_input, canonical)
                # SOP/POS stats = SAME as network stats (by passing the expression string)
                sop_pos_stats = summarize_expr(sop_pos_expr) if sop_pos_expr else summarize_expr("")

                # Network stats from the inner_representation (full_expr)
                net_stats = summarize_expr(full_expr) if full_expr else summarize_expr("")

                # Compare using DAG-preferred multilevel cost
                best_impl = "network" if net_stats["cost_tuple"] < sop_pos_stats["cost_tuple"] else "SOP/POS"

                # min_arch = preferred SOP/POS terms count (first element of preferred (terms, literals))
                min_arch_val = _load_preferred_terms_count(best_json, n_input, canonical)

                writer.writerow({
                    "n_input": n_input,
                    "canonical": canonical,
                    "min_architecture": json.dumps(arch),
                    "hidden_expressions": json.dumps(hidden_exprs, ensure_ascii=False),
                    "output_expression": output_expr,
                    "inner_representation": full_expr,
                    "SOP/POS": sop_pos_expr,

                    # NEW: only the requested stats (no extra SOP/POS fields)
                    "network_stats": json.dumps(net_stats, ensure_ascii=False),
                    "sop_pos_stats": json.dumps(sop_pos_stats, ensure_ascii=False),
                    "best_impl": best_impl,
                    "min_arch": "" if min_arch_val is None else int(min_arch_val),
                })

            except Exception as e:
                writer.writerow({
                    "n_input": n_input,
                    "canonical": row.get("canonical", ""),
                    "min_architecture": json.dumps(_parse_arch(row.get("min_architecture", []))),
                    "hidden_expressions": "",
                    "output_expression": "",
                    "inner_representation": f"ERROR: {type(e).__name__}: {e}",
                    "SOP/POS": "",
                    "network_stats": json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False),
                    "sop_pos_stats": json.dumps({}, ensure_ascii=False),
                    "best_impl": "",
                    "min_arch": "",
                })

# ─────────────────────────────────────────────────────────────────────────────
#  CSV helpers
# ─────────────────────────────────────────────────────────────────────────────
def _hp_from_row(row: pd.Series) -> Hyper:
    lr   = float(row.get("lr", 0.002))
    bs   = row.get("batch_size", None)
    clip = row.get("clip", 0.8)
    mtyp = str(row.get("model_type", "binarized"))

    bs   = int(bs)   if pd.notna(bs)   else None
    clip = float(clip) if pd.notna(clip) else 0.8

    return Hyper(
        lr       = lr,
        batch_size = bs,
        model_type = mtyp,
        clip       = clip,
        attempts   = int(row.get("attempts", 50)  or 50),
        epochs     = int(row.get("epochs",   10_000) or 10_000),
        delta      = float(row.get("delta", 0.01) or 0.01),
        patience   = int(row.get("patience", 100) or 100),
    )

def _parse_arch(val: Any) -> List[int]:
    if isinstance(val, list):
        return [int(x) for x in val]
    if pd.isna(val):
        return []
    try:
        parsed = ast.literal_eval(str(val))
        if isinstance(parsed, list):
            return [int(x) for x in parsed]
        if isinstance(parsed, int):
            return [int(parsed)]
    except Exception:
        pass
    return []

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    Path(Path(OUTPUT_CSV).parent).mkdir(parents=True, exist_ok=True)
    process_rows(INPUT_CSV, BEST_FORM_JSON, OUTPUT_CSV, N_INPUT)
    print(f"Done. Wrote: {OUTPUT_CSV}")

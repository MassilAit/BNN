# COMPLETE SCRIPT — set N_INPUT and file paths below.
# Reads a CSV of successful runs, retrains each model with the row's HP+seed,
# extracts inner representations, and writes them with the preferred SOP/POS
# expression (from a JSON) to an output CSV.

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any
import pandas as pd
import ast
import json
import csv
from pathlib import Path
from find_logic import minimise_one_ones

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model   import ContiniousdModel, BinarizedModel
from dataset import LogicDataSet
from train   import train_model

# ─────────────────────────────────────────────────────────────────────────────
#  USER CONFIG
# ─────────────────────────────────────────────────────────────────────────────
N_INPUT: int = 4  # <<< SET THIS
MODEL_TYPE : str = "continuous" #"binarized" 
INPUT_CSV      = f"{N_INPUT}_{MODEL_TYPE}_output.csv"            # <<< SET IF NEEDED
BEST_FORM_JSON = "npn_classes_bool.json"          # <<< SET IF NEEDED
OUTPUT_CSV     = f"Result/{N_INPUT}_{MODEL_TYPE}.csv"  # <<< SET IF NEEDED

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

    #print(f"Output true : {output_true}")
    #print(f"Output false : {output_false}")
    #print(f"don't cares : {dont_cares}")
    #print(f"hidden min_terms : {hidden_minterms}")

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

    if acc==100.0:
        return analyse_model(model, n_input, dataset, arch)
    
    print(f"Error for {canonical}")

    return "","",""

# ─────────────────────────────────────────────────────────────────────────────
#  CSV/JSON helpers
# ─────────────────────────────────────────────────────────────────────────────
def _hp_from_row(row: pd.Series) -> Hyper:
    lr   = float(row.get("lr", 0.002))
    bs   = row.get("batch_size", None)
    clip = row.get("clip", 0.8)
    mtyp = str(row.get("model_type", "binarized"))

    # NaN handling
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

def _load_preferred_terms(best_json_path: str, n_input: int, canonical: int) -> str:
    """
    Return the preferred terms string (SOP_terms or POS_terms) for the
    Boolean function (n_input, canonical).  If not found → "".
    """
    with open(best_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nk, ck = str(n_input), str(canonical)
    if nk in data and ck in data[nk]:
        entry = data[nk][ck]
        return str(entry.get("POS_terms" if entry.get("preferred", "SOP") == "POS" else "SOP_terms", ""))
    return ""

# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def process_rows(input_csv: str, best_json: str, output_csv: str, n_input: int) -> None:
    df = pd.read_csv(input_csv)
    if "success" in df.columns:
        df = df[df["success"] == True].copy()

    out_fields = [
        "n_input", "canonical", "min_architecture",
        "hidden_expressions", "output_expression",
        "inner_representation", "SOP/POS"        # single column
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

                # normalise result shape
                if isinstance(result, tuple):
                    hidden_exprs, output_expr, full_expr = result
                else:  # no hidden layer or training error
                    hidden_exprs, output_expr, full_expr = [], str(result), str(result)

                writer.writerow({
                    "n_input": n_input,
                    "canonical": canonical,
                    "min_architecture": json.dumps(arch),
                    "hidden_expressions": json.dumps(hidden_exprs, ensure_ascii=False),
                    "output_expression": output_expr,
                    "inner_representation": full_expr,
                    "SOP/POS": _load_preferred_terms(best_json, n_input, canonical),
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
                })

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    process_rows(INPUT_CSV, BEST_FORM_JSON, OUTPUT_CSV, N_INPUT)
    print(f"Done. Wrote: {OUTPUT_CSV}")

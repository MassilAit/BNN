import os

# ─────────────────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
import pandas as pd
import ast
import json
import csv
from pathlib import Path
import time
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
#  Hyper-parameter container
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Hyper:
    """Bundle of all tunable parameters."""
    lr:            float  = 0.002       # Adam learning rate
    delta:         float  = 0.01        # Early-stopping min-Δ
    patience:      int    = 100         # Early-stopping patience
    attempts:      int    = 50          # Re-trainings per architecture
    epochs:        int    = 10_000      # Max epochs per training
    batch_size:    Optional[int] = None # If None → (2**n_input)//4
    model_type:    str    = "binarized" # 'continuous' | 'binarized'
    clip:          float  = 0.8         # clipping for STE (binarized only)

    def to_path_suffix(self) -> str:
        bits = [
            f"model={self.model_type}",
            f"lr={self.lr:.0e}",
            f"delta={self.delta:.0e}",
            f"pat={self.patience}",
            f"att={self.attempts}",
            f"ep={self.epochs}",
            f"clip={self.clip}",
            f"bs={self.batch_size if self.batch_size else 'auto'}"
        ]
        return "__".join(bits).replace('.', 'p')  # filesystem-friendly

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
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=g
    )

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
) -> float:
    """Train one model and return its accuracy on *train_ds*."""
    set_seed(seed)

    if hp.model_type == "continuous":
        model = ContiniousdModel(n_input, hidden_layers, bias=True)
    else:
        model = BinarizedModel(n_input, hidden_layers, bias=True, alpha=hp.clip)

    batch = hp.batch_size if hp.batch_size is not None else max(1, (2**n_input)//4)
    loader = make_loader(train_ds, batch, seed)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = BCEPlusMinusOne()

    train_model(model, loader, optimizer,
                hp.epochs, criterion,
                early_stop=True, delta=hp.delta, patience=hp.patience)

    return evaluate_model_accuracy(train_ds, model), model

def substitute_hidden(expr: str, hidden_exprs: list[str]) -> str:
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

def binary_list_to_int(bits):
    """
    bits: list[int] of 0/1
    returns: integer after reversing the list
    e.g., [1,0] → [0,1] → 1
    """
    bits=bits[::-1]
    value = 0
    for i, b in enumerate(bits):
        value |= (b << i)
    return value
# ───────────────────────────────────────── analyse whole network ──
def analyse_model(model: nn.Module, n_inputs: int, eval_ds: LogicDataSet, n_hidden):
  
    if len(n_hidden)==0 :
        minterms=[]
        with torch.no_grad():
            for i, (X, _) in enumerate(eval_ds):
                # predict output
                y_pred = torch.sign(model(X.unsqueeze(0))).item()
                if y_pred >=0 :
                    minterms.append(i)
        return minimise_one_ones(minterms, n_inputs)

            
    hidden_minterms = [[] for _ in range(n_hidden[0])]      
    output_true = set()
    output_false = set()  
    full_set = set(range(2**n_hidden[0]))                              

    with torch.no_grad():
        for i, (X, _) in enumerate(eval_ds):
            # predict output
            y_pred = torch.sign(model(X.unsqueeze(0))).item()
            #y_pred = 1 if y_pred > 0 else 0             # map -1/+1 → 0/1

            # hidden layer activations
            activ = model.get_activations()["act_0"][0].tolist()
            activ = [1 if a >= 0 else 0 for a in activ]  # map -1/+1 → 0/1

            # record minterms for hidden neurons
            for h, h_val in enumerate(activ):
                if h_val ==1:
                    hidden_minterms[h].append(i)

            if y_pred ==1.0:
                output_true.add(binary_list_to_int(activ))
            else: 
                output_false.add(binary_list_to_int(activ))

 

    dont_cares = full_set - (output_true | output_false)


    output_expression = minimise_one_ones(list(output_true), n_hidden[0], list(dont_cares))


    l = []
    for minterms in hidden_minterms:

        l.append(minimise_one_ones(minterms, n_inputs))

    return substitute_hidden(output_expression, l)
# ─────────────────────────────────────────────────────────────────────────────
def find_min_architecture_for_function(
    n_input: int,
    canonical: int,
    arch: List[int],
    seed : int,
    hp: Hyper,
) -> Tuple[int, Optional[List[int]], Optional[int], float, bool]:
    """
    Returns: (canonical, min_architecture or None, seed or None, accuracy, success_flag)
    Stops at the first (arch, seed) that reaches 100% accuracy (±EPS).
    """
    dataset = LogicDataSet(n_input, canonical)

    acc, model = try_training(n_input, arch, dataset, hp, seed)







# ─────────────────────────────────────────────────────────────────────────────
#  Example usage (Windows-safe)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pass

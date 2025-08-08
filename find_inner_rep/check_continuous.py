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



@dataclass
class Hyper:
    """Bundle of all tunable parameters."""
    lr:            float  = 0.006       # Adam learning rate
    delta:         float  = 0.01        # Early-stopping min-Δ
    patience:      int    = 100         # Early-stopping patience
    attempts:      int    = 458          # Re-trainings per architecture (unused here; single run)
    epochs:        int    = 10_000      # Max epochs per training
    batch_size:    Optional[int] = 8 # If None → (2**n_input)//4
    model_type:    str    = "continuous" # 'continuous' | 'binarized'
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



hp=Hyper()
dataset = LogicDataSet(3, 105)
accuracy, model = try_training(3,[2],dataset,hp, 21)

with torch.no_grad():
        for i, (X, Y) in enumerate(dataset):
            y_pred = torch.sign(model(X.unsqueeze(0))).item()

            print(X)
            # hidden layer activations
            print(model.get_activations())
            
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)
            


            


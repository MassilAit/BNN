# ─────────────────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────────────────
import json
import os
from collections import OrderedDict
from dataclasses import dataclass, asdict
from multiprocessing import Process, cpu_count
from typing import List, Dict

import torch
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
    lr:            float  = 0.005       # Adam learning rate
    delta:         float  = 0.01       # Early-stopping min-Δ
    patience:      int    = 100         # Early-stopping patience
    attempts:      int    = 50          # Re-trainings per architecture
    epochs:        int    = 10_000      # Max epochs per training
    batch_size:    int | None = None    # If None → (2**n_input)//4
    model_type:    str    = "binarized"  # 'continuous' | 'binarized'
    success_ratio: float  = 0.70        # Fraction of runs that must be perfect

    def to_path_suffix(self) -> str:
        """Generate a filesystem-safe string encoding the run's hyperparameters."""
        bits = [
            f"model={self.model_type}",
            f"lr={self.lr:.0e}",
            f"delta={self.delta:.0e}",
            f"pat={self.patience}",
            f"att={self.attempts}",
            f"ep={self.epochs}",
            f"sr={int(self.success_ratio * 100)}",
            f"bs={self.batch_size if self.batch_size else 'auto'}"
        ]
        return "__".join(bits).replace('.', 'p')  # filesystem-friendly

# ─────────────────────────────────────────────────────────────────────────────
#  Loss & evaluation helpers
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

    Notes
    -----
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
#  Training wrappers
# ─────────────────────────────────────────────────────────────────────────────
def try_training(
    n_input: int,
    hidden_layers: List[int],
    loader: DataLoader,
    eval_ds: LogicDataSet,
    hp: Hyper,
    tt_int: int
) -> bool:
    """
    Train one model and return its accuracy on *eval_ds*.
    """
    Model = ContiniousdModel if hp.model_type == "continuous" else BinarizedModel
    model = Model(n_input, hidden_layers, bias=True)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = BCEPlusMinusOne()

    train_model(model, loader, optimizer,
                hp.epochs, criterion,
                early_stop=True, delta=hp.delta, patience=hp.patience)

    if evaluate_model_accuracy(eval_ds, model)==100.0:
        save_path = f"Result/{n_input}_inputs/{tt_int}_weights.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        #compute_logic_function(model)
        return True
    
    return False



def try_multiple_trainings(
    n_input: int,
    hidden_layers: List[int],
    loader: DataLoader,
    eval_ds: LogicDataSet,
    hp: Hyper,
    tt_int: int
)-> None:
    """
    Train the same architecture *hp.attempts* times; return list of accuracies.
    """
    
    for i in range(hp.attempts):
        print(f"Attempt {i}")
        if try_training(n_input, hidden_layers, loader, eval_ds, hp, tt_int):
            return 
        
def train_tes():
    hp=Hyper()
    dataset    = LogicDataSet(3, 6)
    batch_size =  (1 << 2) // 4
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    try_multiple_trainings(3,[3],loader,dataset,hp,6)


train_tes()

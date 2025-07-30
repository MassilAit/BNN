# ─────────────────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────────────────

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Dict
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import json

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
    lr:            float  = 0.002       # Adam learning rate
    delta:         float  = 0.01       # Early-stopping min-Δ
    patience:      int    = 100         # Early-stopping patience
    attempts:      int    = 30          # Re-trainings per architecture
    epochs:        int    = 10_000      # Max epochs per training
    batch_size:    int | None = None    # If None → (2**n_input)//4
    model_type:    str    = "binarized"  # 'continuous' | 'binarized'
    clip:          float  = 0.8           #clippng for STE

    def to_path_suffix(self) -> str:
        """Generate a filesystem-safe string encoding the run's hyperparameters."""
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
#  Seed generation
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """
    Seed RNGs so that parameter initialization is identical
    every time this is called before model construction.
    """
    import random, numpy as np

    random.seed(seed)         # if any Python-random init is used
    np.random.seed(seed)      # if your init uses NumPy
    torch.manual_seed(seed)   # PyTorch params init


def make_loader(dataset, batch_size, seed):
    g = torch.Generator()
    g.manual_seed(seed)  # fixes the shuffle order across runs
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,     # keep 0 for simple reproducibility
        generator=g
    )
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
    train_ds: LogicDataSet,
    hp: Hyper,
    seed : int
) -> float:
    """
    Train one model and return its accuracy on *train_ds*.
    """

    set_seed(seed)
    
    if hp.model_type == "continuous":
        model = ContiniousdModel(n_input, hidden_layers, bias=True)
    else :
        model = BinarizedModel(n_input, hidden_layers, bias=True, alpha = hp.clip)
    

    loader = make_loader(train_ds, hp.batch_size, seed)

    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = BCEPlusMinusOne()

    train_model(model, loader, optimizer,
                hp.epochs, criterion,
                early_stop=True, delta=hp.delta, patience=hp.patience)

    return evaluate_model_accuracy(train_ds, model)


def try_multiple_trainings(
    n_input: int,
    hidden_layers: List[int],
    train_ds: LogicDataSet,
    hp: Hyper
) -> List[float]:
    """
    Train the same architecture *hp.attempts* times; return list of accuracies.
    """

    l=[]
    for i in range(hp.attempts):
        l.append(try_training(n_input, hidden_layers, train_ds, hp, i))
    
    return l

# ─────────────────────────────────────────────────────────────────────────────
#  Architecture sweep for one canonical function
# ─────────────────────────────────────────────────────────────────────────────
def test_architecture(
    n_input: int,
    canonical: int,
    arch : List[int],
    model_type : str,
    lrs : List[float],
    batch_sizes : List[int],
    clips : List[float]
) -> Dict[str, List[float]]:
    """
    Probe increasing architectures until *hp.success_ratio* of runs are perfect.

    Returns
    -------
    dict
        {str(architecture): [acc1, acc2, ...]}
    """
    dataset = LogicDataSet(n_input, canonical)

    stats: Dict[str, List[float]] = OrderedDict()

    def record(arch: List[int], hp:Hyper) -> List[float]:
        accs = try_multiple_trainings(n_input, arch, dataset, hp)
        id=f"lr={hp.lr}, bs={hp.batch_size}, clip={hp.clip}"
        stats[id] = accs
        return accs
    
    if model_type=="continuous":
        for lr, bs in product(lrs, batch_sizes):
            print(f"Testing : lr={lr}, bs={bs}")
            hp=Hyper(lr=lr, batch_size=bs, model_type="continuous")
    
            record(arch, hp)
    else :
        for lr, bs, clip in product(lrs, batch_sizes, clips):
            print(f"Testing : lr={lr}, bs={bs}, clip={clip}")
            hp=Hyper(lr=lr, batch_size=bs, clip=clip, model_type="binarized")
    
            record(arch, hp)

    out_path = os.path.join("Result", f"n={n_input},out={canonical},arch={arch},model={model_type}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats  


# ─────────────────────────────────────────────────────────────────────────────
#  Parallel worker (top-level so it can be pickled by multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────
def _run_one(n_input, canonical, arch, model_type, lrs, clips):
    """
    Worker that prepares batch sizes and calls test_architecture for one (arch, model_type).
    """
    o = 2 ** n_input
    batch_sizes = [o // 4, o // 2, o]
    print(f"Testing model {n_input}, {canonical}, {arch} ({model_type})")
    return test_architecture(n_input, canonical, arch, model_type, lrs, batch_sizes, clips)

# ─────────────────────────────────────────────────────────────────────────────
#  Architecture sweep for all canonical functions (parallelized)
# ─────────────────────────────────────────────────────────────────────────────
def test_all():
    architectures = [
        (3, 30, [3]),
        (3, 30, [4]),
        (4, 426, [3]),
        (4, 426, [4]),
        (4, 1969, [2]),
        (4, 1969, [3])
    ]

    lrs   = [0.004, 0.005, 0.006]
    clips = [0.6, 0.8, 0.9]

    # Build all tasks (continuous + binarized for each architecture)
    tasks = []
    for n_input, canonical, arch in architectures:
        tasks.append((n_input, canonical, arch, "continuous", lrs, clips))
        tasks.append((n_input, canonical, arch, "binarized",  lrs, clips))

    # Run in parallel
    # Tip: set max_workers to control parallelism, e.g. ProcessPoolExecutor(max_workers=os.cpu_count()-1)
    with ProcessPoolExecutor() as ex:
        futures = [ex.submit(_run_one, *t) for t in tasks]
        for fut in as_completed(futures):
            try:
                _ = fut.result()  # already saved to disk by test_architecture
            except Exception as e:
                print(f"[ERROR] A task failed: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  Entry point (required on Windows; recommended for PyTorch)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Use 'spawn' to avoid issues with PyTorch/NumPy state in child processes
    mp.set_start_method("spawn", force=True)
    test_all()

        
   













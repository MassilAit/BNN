import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ─────────────────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────────────────
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import pandas as pd
import ast
import json
import csv
from pathlib import Path
import time

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
    lr:            float  = 0.005       # Adam learning rate
    delta:         float  = 0.01        # Early-stopping min-Δ
    patience:      int    = 100         # Early-stopping patience
    attempts:      int    = 10          # Re-trainings per architecture
    epochs:        int    = 10_000      # Max epochs per training
    batch_size:    int | None = None    # If None → (2**n_input)//4
    model_type:    str    = "continuous" # 'continuous' | 'binarized'
    clip:          float  = 0.8         # clipping for STE (used if binarized)

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
        num_workers=0,     # keep 0 for reproducibility
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
#  Training wrappers
# ─────────────────────────────────────────────────────────────────────────────
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

    return evaluate_model_accuracy(train_ds, model)

def try_multiple_trainings(
    n_input: int,
    hidden_layers: List[int],
    train_ds: LogicDataSet,
    hp: Hyper
) -> List[float]:
    """Train the same architecture *hp.attempts* times; return list of accuracies."""
    return [try_training(n_input, hidden_layers, train_ds, hp, i) for i in range(hp.attempts)]

# ─────────────────────────────────────────────────────────────────────────────
#  Selection helpers
# ─────────────────────────────────────────────────────────────────────────────
EPS = 1e-9  # near-100% tolerance

def _succ_rate(accs: List[float]) -> float:
    return sum(a >= 100.0 - EPS for a in accs) / max(1, len(accs))

def _mean_acc(accs: List[float]) -> float:
    return sum(accs) / max(1, len(accs))

# ─────────────────────────────────────────────────────────────────────────────
#  LR×BS sweep for a single (canonical, architecture)
# ─────────────────────────────────────────────────────────────────────────────
def sweep_lr_bs(
    n_input: int,
    canonical: int,
    arch: List[int],
    model_type: str,
    lrs: List[float],
    bss: List[int],
    attempts: int = 10,
    epochs: int = 10_000,
    delta: float = 0.01,
    patience: int = 100,
    clip: float = 0.8,   # used if binarized
) -> Tuple[int, List[int], float, int, float]:
    """
    Return:
      (canonical, arch, best_lr, best_bs, success_rate)
    """
    dataset = LogicDataSet(n_input, canonical)
    stats: Dict[Tuple[float, int], List[float]] = OrderedDict()

    for lr in lrs:
        for bs in bss:
            hp = Hyper(
                lr=lr,
                batch_size=bs,
                model_type=model_type,
                clip=clip,
                attempts=attempts,
                epochs=epochs,
                delta=delta,
                patience=patience,
            )
            accs = try_multiple_trainings(n_input, arch, dataset, hp)
            stats[(lr, bs)] = accs

    # Choose best: success rate → mean acc → smaller bs → lower lr
    def score_key(k: Tuple[float, int]):
        lr, bs = k
        accs = stats[k]
        return (_succ_rate(accs), _mean_acc(accs), bs, -lr)

    best_lr, best_bs = max(stats.keys(), key=score_key)
    best_succ = _succ_rate(stats[(best_lr, best_bs)])
    return canonical, arch, float(best_lr), int(best_bs), float(best_succ)

# ─────────────────────────────────────────────────────────────────────────────
#  Per-task function (top-level for Windows pickling)
# ─────────────────────────────────────────────────────────────────────────────
def run_one(n_input: int,
            canonical: int,
            arch: List[int],
            model_type: str,
            lrs: List[float],
            bss: List[int]) -> Tuple[int, List[int], float, int, float]:
    """
    Run one (canonical, arch) and return:
      (canonical, arch, best_lr, best_bs, success_rate)
    """
    print(f"Testing model n={n_input}, canonical={canonical}, arch={arch} ({model_type})")
    return sweep_lr_bs(
        n_input=n_input,
        canonical=canonical,
        arch=arch,
        model_type=model_type,
        lrs=lrs,
        bss=bss,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  CSV helpers: append row safely, resume by skipping existing rows
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_csv_header(path: Path, columns: List[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(columns)

def _append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([row["canonical"], row["architecture"], row["best_lr"], row["best_bs"], row["success_rate"]])
        f.flush()

def _load_existing_keys(path: Path) -> set[Tuple[int, str]]:
    """
    Load existing (canonical, architecture_json_string) to avoid duplicates
    when resuming.
    """
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path)
        keys = set()
        if "canonical" in df.columns and "architecture" in df.columns:
            for _, r in df.iterrows():
                canonical = int(r["canonical"])
                arch_str = str(r["architecture"])
                keys.add((canonical, arch_str))
        return keys
    except Exception:
        return set()

# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator (parallel, incremental CSV writes)
# ─────────────────────────────────────────────────────────────────────────────
def run_all(n_input: int,
            path: str,
            lrs: List[float],
            model_type: str,
            output_csv: str,
            max_workers: int | None = None) -> None:
    """
    Read (canonical, minimal_architecture) CSV, compute best (lr, bs) per row in parallel,
    and append (canonical, architecture, best_lr, best_bs, success_rate) to output_csv
    as each task finishes (crash-safe).
    """
    # 3×3 grid
    lrs = [float(x) for x in lrs]
    n = 2**n_input
    bss = [n//4, n//2, n]

    # Read & parse source CSV
    df = pd.read_csv(path)
    df["canonical"] = df["canonical"].astype(int)
    df["architecture"] = df["architecture"].apply(ast.literal_eval)

    # Build tasks: one per non-empty architecture
    tasks: List[Tuple[int, List[int]]] = [
        (int(row["canonical"]), row["architecture"])
        for _, row in df.iterrows()
        if isinstance(row["architecture"], list) and len(row["architecture"]) > 0
    ]

    out_path = Path(output_csv)
    columns = ["canonical", "architecture", "best_lr", "best_bs", "success_rate"]
    _ensure_csv_header(out_path, columns)

    # Resume support: skip tasks already present
    existing = _load_existing_keys(out_path)
    tasks = [
        (canonical, arch)
        for (canonical, arch) in tasks
        if (canonical, json.dumps(arch)) not in existing
    ]

    if not tasks:
        print(f"No work to do. {output_csv} already has all rows.")
        return

    n_workers = max_workers or mp.cpu_count()
    print(f"Launching {len(tasks)} tasks with {n_workers} workers...")

    # Submit tasks and append results as they complete
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        fut_to_task = {
            ex.submit(run_one, n_input, canonical, arch, model_type, lrs, bss): (canonical, arch)
            for (canonical, arch) in tasks
        }

        for fut in as_completed(fut_to_task):
            canonical, arch = fut_to_task[fut]
            try:
                c, a, best_lr, best_bs, succ_rate = fut.result()
                row = {
                    "canonical": c,
                    "architecture": json.dumps(a),  # store as JSON string
                    "best_lr": best_lr,
                    "best_bs": best_bs,
                    "success_rate": succ_rate,
                }
                _append_csv_row(out_path, row)
                print(f"[OK] canonical={c}, arch={a} → lr={best_lr}, bs={best_bs}, succ={succ_rate:.3f}")
            except Exception as e:
                print(f"[ERROR] canonical={canonical}, arch={arch} → {e}")

    print(f"Appended results to {output_csv}.")

# ─────────────────────────────────────────────────────────────────────────────
#  Example usage (Windows-safe)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    print("Starting time...")
    start_time = time.time()

    run_all(
        n_input=2,
        path="2input_continuous.csv",   # expects columns: canonical, minimal_architecture
        lrs=[0.004, 0.005, 0.006],      # 3 LRs
        model_type="continuous",        # or "binarized" (clip used from Hyper)
        output_csv="2output_continuous.csv",
        max_workers=None,               # None → use all logical cores
    )

    end_time = time.time()
    elapsed = end_time - start_time
    h, rem = divmod(elapsed, 3600)
    m, s  = divmod(rem, 60)
    print(f"Elapsed time: {int(h):02}:{int(m):02}:{s:.2f}")

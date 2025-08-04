import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ─────────────────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────────────────
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional
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

    return evaluate_model_accuracy(train_ds, model)

# ─────────────────────────────────────────────────────────────────────────────
#  CSV helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_csv_header(path: Path, columns: List[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(columns)

def _append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            row["canonical"],
            row["min_architecture"],
            row["seed"],
            row["lr"],
            row["batch_size"],
            row["clip"],
            row["model_type"],
            row["attempts"],
            row["epochs"],
            row["delta"],
            row["patience"],
            row["accuracy"],
            row["success"],
        ])
        f.flush()

def _load_done_canonicals(path: Path) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path)
        if "canonical" in df.columns:
            return set(int(c) for c in df["canonical"].tolist())
    except Exception:
        pass
    return set()

# ─────────────────────────────────────────────────────────────────────────────
#  HP builder from input CSV row
# ─────────────────────────────────────────────────────────────────────────────
def _hp_from_row(row, default_model_type: str,
                 attempts: int, epochs: int, delta: float, patience: int) -> Hyper:
    lr   = float(row.get("best_lr", 0.002))
    bs   = row.get("best_bs", None)
    clip = row.get("best_clip", 0.8)
    mtyp = str(row.get("model_type", default_model_type))

    # Handle NaNs/empties
    try:
        bs = int(bs) if pd.notna(bs) else None
    except Exception:
        bs = None
    try:
        clip = float(clip) if pd.notna(clip) else 0.0
    except Exception:
        clip = 0.0

    return Hyper(
        lr=lr,
        batch_size=bs,
        model_type=mtyp,
        clip=clip,
        attempts=attempts,
        epochs=epochs,
        delta=delta,
        patience=patience,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Core search: minimal architecture per canonical
# ─────────────────────────────────────────────────────────────────────────────
def find_min_architecture_for_function(
    n_input: int,
    canonical: int,
    arch_candidates: List[List[int]],
    hp: Hyper,
) -> Tuple[int, Optional[List[int]], Optional[int], float, bool]:
    """
    Returns: (canonical, min_architecture or None, seed or None, accuracy, success_flag)
    Stops at the first (arch, seed) that reaches 100% accuracy (±EPS).
    """
    dataset = LogicDataSet(n_input, canonical)

    for arch in arch_candidates:
        for seed in range(hp.attempts):
            acc = try_training(n_input, arch, dataset, hp, seed)
            if acc >= 100.0 - EPS:
                return canonical, arch, seed, acc, True

    # If no success, report best observed acc and no seed
    # (Here we just return the last tried arch; you can track global-best if desired)
    return canonical, None, None, acc, False  # acc is from last attempt

# Top-level worker (picklable)
def _worker_min_arch(n_input: int,
                     canonical: int,
                     arch_candidates: List[List[int]],
                     hp_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    hp = Hyper(**hp_kwargs)
    c, arch, seed, acc, ok = find_min_architecture_for_function(
        n_input=n_input,
        canonical=canonical,
        arch_candidates=arch_candidates,
        hp=hp
    )
    return {
        "canonical": c,
        "min_architecture": json.dumps(arch) if arch is not None else "",
        "seed": seed if seed is not None else -1,
        "lr": hp.lr,
        "batch_size": hp.batch_size if hp.batch_size is not None else "auto",
        "clip": hp.clip if hp.model_type == "binarized" else "",
        "model_type": hp.model_type,
        "attempts": hp.attempts,
        "epochs": hp.epochs,
        "delta": hp.delta,
        "patience": hp.patience,
        "accuracy": acc,
        "success": 1 if ok else 0,
    }

# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run_min_arch_search(
    n_input: int,
    hp_csv: str,             # CSV with best HP per function
    output_csv: str,         # Results CSV to append one row per canonical
    arch_min: int = 2,
    arch_max: int = 5,
    default_model_type: str = "binarized",
    max_workers: Optional[int] = None,
    attempts: int = 50,
    epochs: int = 10_000,
    delta: float = 0.01,
    patience: int = 100,
) -> None:

    # Read HP CSV; dedupe per canonical (keep first)
    df_hp = pd.read_csv(hp_csv)
    if "canonical" not in df_hp.columns:
        raise ValueError("Input CSV must contain a 'canonical' column.")

    df_hp["canonical"] = df_hp["canonical"].astype(int)
    df_hp = df_hp.drop_duplicates(subset=["canonical"], keep="first").reset_index(drop=True)

    # Prepare arch candidates: single hidden layer [k] for k in [arch_min..arch_max]
    arch_candidates = [[k] for k in range(arch_min, arch_max + 1)]

    out_path = Path(output_csv)
    columns = [
        "canonical", "min_architecture", "seed",
        "lr", "batch_size", "clip", "model_type",
        "attempts", "epochs", "delta", "patience",
        "accuracy", "success",
    ]
    _ensure_csv_header(out_path, columns)

    # Resume: skip canonicals already done
    done = _load_done_canonicals(out_path)
    rows = df_hp.to_dict(orient="records")
    rows = [r for r in rows if int(r["canonical"]) not in done]

    if not rows:
        print(f"No work to do. {output_csv} already has all rows.")
        return

    n_workers = max_workers or mp.cpu_count()
    print(f"Launching {len(rows)} canonicals with {n_workers} workers...")

    # Build immutable HP kwargs per row
    tasks = []
    for r in rows:
        hp_obj = _hp_from_row(r, default_model_type, attempts, epochs, delta, patience)
        hp_kwargs = dict(
            lr=hp_obj.lr,
            delta=hp_obj.delta,
            patience=hp_obj.patience,
            attempts=hp_obj.attempts,
            epochs=hp_obj.epochs,
            batch_size=hp_obj.batch_size,
            model_type=hp_obj.model_type,
            clip=hp_obj.clip,
        )
        tasks.append((int(r["canonical"]), hp_kwargs))

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        fut_to_canon = {
            ex.submit(_worker_min_arch, n_input, canonical, arch_candidates, hp_kwargs): canonical
            for (canonical, hp_kwargs) in tasks
        }

        for fut in as_completed(fut_to_canon):
            canonical = fut_to_canon[fut]
            try:
                row = fut.result()
                _append_csv_row(out_path, row)
                if row["success"] == 1:
                    print(f"[OK] canonical={canonical} → min_arch={row['min_architecture']}, seed={row['seed']}, acc={row['accuracy']:.1f}")
                else:
                    print(f"[FAIL] canonical={canonical} → no 100% within attempts (last acc={row['accuracy']:.1f})")
            except Exception as e:
                print(f"[ERROR] canonical={canonical} → {e}")

    print(f"Appended results to {output_csv}.")

# ─────────────────────────────────────────────────────────────────────────────
#  Example usage (Windows-safe)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    # === Configure here ===
    N_INPUT        = 3
    HP_CSV         = "3_binarized.csv"   # must have: canonical, best_lr, [best_bs], [best_clip], [model_type]
    OUT_CSV        = "3_binarized_output.csv"
    MODEL_TYPE_DFT = "binarized"                  # used if model_type not present in HP_CSV
    ARCH_MIN, ARCH_MAX = 2, 5

    # You can tune these to trade speed vs reliability
    ATTEMPTS = 458
    EPOCHS   = 10_000
    DELTA    = 0.01
    PATIENCE = 100
    MAX_WORKERS = None  # None → use mp.cpu_count()

    print("Starting time...")
    start_time = time.time()

    run_min_arch_search(
        n_input=N_INPUT,
        hp_csv=HP_CSV,
        output_csv=OUT_CSV,
        arch_min=ARCH_MIN,
        arch_max=ARCH_MAX,
        default_model_type=MODEL_TYPE_DFT,
        max_workers=MAX_WORKERS,
        attempts=ATTEMPTS,
        epochs=EPOCHS,
        delta=DELTA,
        patience=PATIENCE,
    )

    elapsed = time.time() - start_time
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"Elapsed time: {int(h):02}:{int(m):02}:{s:.2f}")

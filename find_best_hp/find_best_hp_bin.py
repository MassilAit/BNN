import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ─────────────────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────────────────
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
    lr:            float  = 0.002
    delta:         float  = 0.01
    patience:      int    = 100
    attempts:      int    = 10
    epochs:        int    = 10_000
    batch_size:    int | None = None
    model_type:    str    = "binarized"  # 'continuous' | 'binarized'
    clip:          float  = 0.8          # STE clipping (binarized only)

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
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=g)

# ─────────────────────────────────────────────────────────────────────────────
#  Loss & evaluation
# ─────────────────────────────────────────────────────────────────────────────
class BCEPlusMinusOne(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    def forward(self, logits, targets_pm1):
        targets_01 = (targets_pm1 + 1) / 2
        return self.bce(logits, targets_01)

def evaluate_model_accuracy(dataset: LogicDataSet, model: nn.Module) -> float:
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
def try_training(n_input: int, hidden_layers: List[int], train_ds: LogicDataSet, hp: Hyper, seed: int) -> float:
    set_seed(seed)
    if hp.model_type == "continuous":
        model = ContiniousdModel(n_input, hidden_layers, bias=True)
    else:
        model = BinarizedModel(n_input, hidden_layers, bias=True, alpha=hp.clip)
    batch = hp.batch_size if hp.batch_size is not None else max(1, (2**n_input)//4)
    loader = make_loader(train_ds, batch, seed)
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    criterion = BCEPlusMinusOne()
    train_model(model, loader, optimizer, hp.epochs, criterion, early_stop=True, delta=hp.delta, patience=hp.patience)
    return evaluate_model_accuracy(train_ds, model)

def try_multiple_trainings(n_input: int, hidden_layers: List[int], train_ds: LogicDataSet, hp: Hyper) -> List[float]:
    return [try_training(n_input, hidden_layers, train_ds, hp, i) for i in range(hp.attempts)]

# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────
EPS = 1e-9
def _succ_rate(accs: List[float]) -> float:
    return sum(a >= 100.0 - EPS for a in accs) / max(1, len(accs))

def _mean_acc(accs: List[float]) -> float:
    return sum(accs) / max(1, len(accs))

# ─────────────────────────────────────────────────────────────────────────────
#  LR×BS×CLIP sweep → best only
# ─────────────────────────────────────────────────────────────────────────────
def sweep_lr_bs_clip(
    n_input: int,
    canonical: int,
    arch: List[int],
    model_type: str,
    lrs: List[float],
    batch_sizes: List[int],
    clip_values: List[float],
    attempts: int = 10,
    epochs: int = 10_000,
    delta: float = 0.01,
    patience: int = 100,
) -> List[Dict[str, Any]]:
    """
    Returns ONE row (best lr, bs, clip) for this (canonical, architecture):
      [{'canonical','architecture','best_lr','best_bs','best_clip','success_rate'}]
    """
    dataset = LogicDataSet(n_input, canonical)
    # For binarized we use provided clip_values; for continuous you'd set [0.0]

    stats: Dict[Tuple[float, int, float], List[float]] = {}

    # Evaluate all combos
    for lr in lrs:
        for bs in batch_sizes:
            for clip in clip_values:
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
                stats[(float(lr), int(bs), float(clip))] = accs

    # Pick best: success rate → mean acc → smaller bs → lower clip → lower lr
    def score_key(k: Tuple[float, int, float]):
        lr, bs, clip = k
        accs = stats[k]
        return (_succ_rate(accs), _mean_acc(accs), -bs, -clip, -lr)

    best_lr, best_bs, best_clip = max(stats.keys(), key=score_key)
    best_sr = _succ_rate(stats[(best_lr, best_bs, best_clip)])

    return [{
        "canonical": int(canonical),
        "architecture": json.dumps(arch),
        "best_lr": float(best_lr),
        "best_bs": int(best_bs),
        "best_clip": float(best_clip),
        "success_rate": float(best_sr),
    }]

# ─────────────────────────────────────────────────────────────────────────────
#  CSV helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_csv_header(path: Path, columns: List[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(columns)

def _append_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow([r["canonical"], r["architecture"], r["best_lr"], r["best_bs"], r["best_clip"], r["success_rate"]])
        f.flush()

def _load_existing_pairs(path: Path) -> set[Tuple[int, str]]:
    """
    Keys are (canonical, architecture_json) so each pair is written once.
    """
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path)
        need = {"canonical","architecture"}
        if not need.issubset(set(df.columns)):
            return set()
        keys: set[Tuple[int, str]] = set()
        for _, r in df.iterrows():
            keys.add((int(r["canonical"]), str(r["architecture"])))
        return keys
    except Exception:
        return set()

# ─────────────────────────────────────────────────────────────────────────────
#  TOP-LEVEL WORKER
# ─────────────────────────────────────────────────────────────────────────────
def _run_one_lr_bs_clip(n_input, model_type, lrs, batch_sizes, clip_values,
                        attempts, epochs, delta, patience,
                        canonical, arch) -> List[Dict[str, Any]]:
    if not arch:
        return []
    return sweep_lr_bs_clip(
        n_input=n_input,
        canonical=canonical,
        arch=arch,
        model_type=model_type,
        lrs=lrs,
        batch_sizes=batch_sizes,
        clip_values=clip_values,
        attempts=attempts,
        epochs=epochs,
        delta=delta,
        patience=patience,
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def run_all_lr_bs_clip(
    n_input: int,
    input_csv: str,               # contains columns: canonical, architecture
    lrs: List[float],
    batch_sizes: List[int],
    clip_values: List[float],
    model_type: str,              # "binarized"
    output_csv: str,
    max_workers: int | None = None,
    attempts: int = 10,
    epochs: int = 10_000,
    delta: float = 0.01,
    patience: int = 100,
) -> None:

    # Read input CSV: only canonical & architecture are required
    df_in = pd.read_csv(input_csv)
    if "canonical" not in df_in.columns or "architecture" not in df_in.columns:
        raise ValueError("Input CSV must have columns: canonical, architecture")

    # Parse rows (skip empty architectures)
    tasks: List[Tuple[int, List[int]]] = []
    for _, row in df_in.iterrows():
        c = int(row["canonical"])
        a = row["architecture"]
        if isinstance(a, float) and pd.isna(a):
            continue
        if isinstance(a, str):
            s = a.strip()
            if s in ("", "[]"):
                continue
            try:
                a = ast.literal_eval(s)
            except Exception:
                try:
                    a = [int(s)]
                except Exception:
                    continue
        if isinstance(a, (int, float)):
            a = [int(a)]
        if not (isinstance(a, list) and all(isinstance(x, (int, float)) for x in a)):
            continue
        if len(a) == 0:
            continue
        a = [int(x) for x in a]
        tasks.append((c, a))

    out_path = Path(output_csv)
    columns = ["canonical", "architecture", "best_lr", "best_bs", "best_clip", "success_rate"]
    _ensure_csv_header(out_path, columns)

    # Resume support: skip (canonical, architecture) pairs already done
    existing_pairs = _load_existing_pairs(out_path)
    tasks = [
        (c, a) for (c, a) in tasks
        if (c, json.dumps(a)) not in existing_pairs
    ]

    if not tasks:
        print(f"No work to do. {output_csv} already has all rows.")
        return

    n_workers = max_workers or mp.cpu_count()
    print(f"Launching {len(tasks)} (canonical,arch) jobs with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        fut_to_task = {
            ex.submit(
                _run_one_lr_bs_clip,
                n_input, model_type, lrs, batch_sizes, clip_values,
                attempts, epochs, delta, patience,
                c, list(a)
            ): (c, list(a))
            for (c, a) in tasks
        }

        for fut in as_completed(fut_to_task):
            c, a = fut_to_task[fut]
            try:
                rows = fut.result()  # one row list
                if rows:
                    _append_csv_rows(out_path, rows)
                    existing_pairs.add((c, json.dumps(a)))
                    print(f"[OK] canonical={c}, arch={a} → best written")
                else:
                    print(f"[SKIP] canonical={c}, arch={a} → empty result")
            except Exception as e:
                print(f"[ERROR] canonical={c}, arch={a} → {e}")

    print(f"Appended results to {output_csv}.")

# ─────────────────────────────────────────────────────────────────────────────
#  Example usage (Windows-safe)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # === Configure here ===
    N_INPUT      = 3
    n            = 2**N_INPUT
    INPUT_CSV    = "3input_binarized.csv"  # must contain: canonical, architecture
    MODEL_TYPE   = "binarized"             # binarized search over clip

    LRS          = [0.004, 0.005, 0.006]
    BATCH_SIZES  = [n//4, n//2, n]
    CLIP_VALUES  = [0.6, 0.8, 0.9]
    OUT_CSV      = "3output_binarized.csv"
    MAX_WORKERS  = None

    # You can lower attempts/epochs to speed up runs
    ATTEMPTS = 10
    EPOCHS   = 10_000
    DELTA    = 0.01
    PATIENCE = 100

    print("Starting time...")
    start_time = time.time()

    run_all_lr_bs_clip(
        n_input=N_INPUT,
        input_csv=INPUT_CSV,
        lrs=LRS,
        batch_sizes=BATCH_SIZES,
        clip_values=CLIP_VALUES,
        model_type=MODEL_TYPE,
        output_csv=OUT_CSV,
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

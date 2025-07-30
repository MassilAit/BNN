# ─────────────────────────────────────────────────────────────────────────────
#  Imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
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
    lr:            float  = 0.002       # Adam learning rate
    delta:         float  = 0.01       # Early-stopping min-Δ
    patience:      int    = 100         # Early-stopping patience
    attempts:      int    = 10          # Re-trainings per architecture
    epochs:        int    = 10_000      # Max epochs per training
    batch_size:    int | None = None    # If None → (2**n_input)//4
    model_type:    str    = "continuous"  # 'continuous' | 'binarized'
    success_ratio: float  = 0.30        # Fraction of runs that must be perfect

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
    hp: Hyper
) -> float:
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

    return evaluate_model_accuracy(eval_ds, model)


def try_multiple_trainings(
    n_input: int,
    hidden_layers: List[int],
    loader: DataLoader,
    eval_ds: LogicDataSet,
    hp: Hyper
) -> List[float]:
    """
    Train the same architecture *hp.attempts* times; return list of accuracies.
    """
    return [
        try_training(n_input, hidden_layers, loader, eval_ds, hp)
        for _ in range(hp.attempts)
    ]

# ─────────────────────────────────────────────────────────────────────────────
#  Architecture sweep for one canonical function
# ─────────────────────────────────────────────────────────────────────────────
def find_architecture_stats(
    n_input: int,
    canonical: str,
    hp: Hyper,
    max_factor: int = 3
) -> Dict[str, List[float]]:
    """
    Probe increasing architectures until *hp.success_ratio* of runs are perfect.

    Returns
    -------
    dict
        {str(architecture): [acc1, acc2, ...]}
    """
    dataset    = LogicDataSet(n_input, canonical)
    batch_size = hp.batch_size or (1 << n_input) // 4
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    stats: Dict[str, List[float]] = OrderedDict()

    def record(arch: List[int]) -> List[float]:
        accs = try_multiple_trainings(n_input, arch, loader, dataset, hp)
        stats[str(arch)] = accs
        return accs

    # 0-hidden
    if sum(a == 100.0 for a in record([])) / hp.attempts >= hp.success_ratio:
        return stats

    # 1-hidden
    for width in range(2, (1 << n_input) + 1):
        if sum(a == 100.0 for a in record([width])) / hp.attempts >= hp.success_ratio:
            break

    # 2-hidden (total neurons limited)
    #for total in range(4, max_factor * n_input + 1):
    #    for h1 in range(2, total - 1):
    #        h2 = total - h1
    #        if h2 <= h1:
    #            if sum(a == 100.0 for a in record([h1, h2])) / hp.attempts >= hp.success_ratio:
    #                return stats
    return stats  # may be empty

# ─────────────────────────────────────────────────────────────────────────────
#  Multiprocessing
# ─────────────────────────────────────────────────────────────────────────────
def analyze_sublist_json(n_input: int, canonicals: List[str], hp: Hyper, idx: int, path : str):
    """
    Worker: analyse a sub-list and save incremental JSON fragment.
    """
    out_path = os.path.join(path, f"summary_part_{idx}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    result: Dict[str, Dict] = {}
    for i, val in enumerate(canonicals, 1):
        print(f"[{idx}] ({i}/{len(canonicals)}) {val}")
        result[val] = find_architecture_stats(n_input, val, hp)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)

    print(f"[{idx}] done → {out_path}")


def split_list(lst, k):  # round-robin split
    return [lst[i::k] for i in range(k)]


def analyze_all_canonical_forms_parallel(
    n_input: int,
    canonicals: List[str],
    hp: Hyper,
    num_chunks: int | None = None
):
    """
    Spawn *num_chunks* processes, each handling a slice of the canonical list.
    """
    num_chunks = num_chunks or min(cpu_count(), 8)
    num_chunks = min(num_chunks, len(canonicals))
    sublists   = split_list(canonicals, num_chunks)

    suffix = hp.to_path_suffix()
    result_dir = os.path.join("Result", f"{n_input}_inputs__{suffix}")
    os.makedirs(result_dir, exist_ok=True)

    procs = [
        Process(target=analyze_sublist_json, args=(n_input, sub, hp, i, result_dir))
        for i, sub in enumerate(sublists)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    # merge
    merged: Dict[str, Dict] = {}
    for i in range(num_chunks):
        with open(os.path.join(result_dir, f"summary_part_{i}.json")) as f:
            merged.update(json.load(f))

    final_path = os.path.join(result_dir, "summary.json")
    with open(final_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f" Final merged summary saved to {final_path}")

# ─────────────────────────────────────────────────────────────────────────────
#  Top-level driver
# ─────────────────────────────────────────────────────────────────────────────
def analyze_npn_json(
    n_input: int,
    hp: Hyper,
    json_path: str = "npn_classes_brute.json",
    num_chunks: int | None = None
):
    """
    Load canonical list for *n_input* from *json_path* and launch analysis.
    """
    with open(json_path) as f:
        canonicals = json.load(f)[str(n_input)]

    print(f"\n=== {len(canonicals)} functions, n={n_input} ===")
    analyze_all_canonical_forms_parallel(n_input, canonicals, hp,
                                         num_chunks=num_chunks)

# ─────────────────────────────────────────────────────────────────────────────
#  Command-line interface
# ─────────────────────────────────────────────────────────────────────────────
def parse_cli():
    base = asdict(Hyper())          
    p = argparse.ArgumentParser()

    p.add_argument("--n-input", required=True, type=int)
    p.add_argument("--model-type", choices=["continuous", "binarized"], required=True)

    p.add_argument("--learning-rate", type=float, default=base["lr"])
    p.add_argument("--delta",         type=float, default=base["delta"])
    p.add_argument("--patience",      type=int,   default=base["patience"])
    p.add_argument("--attempts",      type=int,   default=base["attempts"])
    p.add_argument("--epochs",        type=int,   default=base["epochs"])
    p.add_argument("--batch-size",    type=int,   default=base["batch_size"])
    p.add_argument("--success-ratio", type=float, default=base["success_ratio"])

    p.add_argument("--json-path", default="npn_classes_brute.json")
    p.add_argument("--num-chunks", type=int, default=None)
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_cli()
    hp = Hyper(lr            = args.learning_rate,
               delta         = args.delta,
               patience      = args.patience,
               attempts      = args.attempts,
               epochs        = args.epochs,
               batch_size    = args.batch_size,
               model_type    = args.model_type,
               success_ratio = args.success_ratio)

    if hp.batch_size is None:
        hp.batch_size = (1 << args.n_input) // 4

    analyze_npn_json(args.n_input, hp,
                     json_path=args.json_path,
                     num_chunks=args.num_chunks)

if __name__ == "__main__":
    main()

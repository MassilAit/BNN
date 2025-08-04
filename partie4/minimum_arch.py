"""
Scan multiple summary.json files, find the smallest architecture that reaches
100 % accuracy for each canonical Boolean function, and save the result to CSV.

Edit INPUT_JSONS and OUTPUT_CSV as needed, then run:

    python min_arch_global.py
"""
import json
import csv
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

# 🔧 1) Put here every summary.json you want to analyse
INPUT_JSONS = [
    "Result/2_inputs__model=continuous__lr=2e-03__delta=1e-02__pat=100__att=10__ep=10000__sr=30__bs=1/summary.json",
    "Result/2_inputs__model=continuous__lr=5e-03__delta=1e-02__pat=100__att=10__ep=10000__sr=30__bs=1/summary.json"
]
# 🔧 2) Output file
OUTPUT_CSV = "2input_continuous.csv"

EPS = 1e-9  # float tolerance for 100.0


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def as_float(x: Any) -> float:
    """Convert possible string/number to float."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise ValueError(f"Cannot convert {x!r} to float.")


def has_perfect(accs: Iterable[Any]) -> bool:
    """True if any accuracy is ≈ 100 % (0–100 scale)."""
    for a in accs:
        try:
            if as_float(a) >= 100.0 - EPS:
                return True
        except Exception:
            # Ignore non-convertible entries
            continue
    return False


def arch_key(arch_name: str) -> Tuple[int, int, List[int]]:
    """
    Ranking key → smaller is better.

    • primary  : total neurons  (sum of layer sizes)
    • secondary: # layers
    • tertiary : layer list itself (lexicographic)
      (keeps deterministic order when sums are equal)
    """
    try:
        # arch string should look like "[4, 4]" or "[6]"
        layers = json.loads(arch_name)
        if isinstance(layers, list) and all(isinstance(x, int) for x in layers):
            return sum(layers), len(layers), layers
    except Exception:
        pass
    # Fallback: put weird formats at the end
    return (1 << 30), 1 << 30, []


# --------------------------------------------------------------------------- #
# Main logic
# --------------------------------------------------------------------------- #
def collect_all_perfect_architectures() -> Dict[str, List[str]]:
    """
    Read every INPUT_JSONS file and store *all* architectures that achieve
    at least one perfect run, grouped by canonical function.
    """
    perfect_archs: Dict[str, List[str]] = defaultdict(list)

    for path in INPUT_JSONS:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"{path}: top-level JSON must map canonical → "
                "{architecture: [accuracies]}"
            )

        for canonical, arch_dict in data.items():
            if not isinstance(arch_dict, dict):
                continue
            for arch_name, acc_list in arch_dict.items():
                if isinstance(acc_list, (list, tuple)) and has_perfect(acc_list):
                    perfect_archs[canonical].append(arch_name)

    return perfect_archs


def select_global_min(perfect_archs: Dict[str, List[str]]) -> Dict[str, str]:
    """Pick the smallest architecture (by arch_key) for each canonical."""
    result = {}
    for canonical, archs in perfect_archs.items():
        if archs:
            result[canonical] = min(archs, key=arch_key)
        else:
            # No architecture ever hit 100 % → leave blank
            result[canonical] = ""
    return result


def main() -> None:
    perfect_archs = collect_all_perfect_architectures()
    min_arch = select_global_min(perfect_archs)

    # --------------------------------------------------------------------- #
    # Write CSV
    # --------------------------------------------------------------------- #
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["canonical", "architecture"])
        for canonical, arch in sorted(min_arch.items()):
            w.writerow([canonical, arch])

    print(f"Wrote {OUTPUT_CSV} with {len(min_arch)} rows.")


if __name__ == "__main__":
    main()

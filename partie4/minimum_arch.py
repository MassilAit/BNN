
import json
import csv
from typing import Any, Dict, Iterable

# 🔧 Edit these two lines:
INPUT_JSON  = "Result/4_inputs__model=continuous__lr=5e-03__delta=1e-02__pat=100__att=10__ep=10000__sr=30__bs=4/summary.json"
OUTPUT_CSV  = "output.csv"

EPS = 1e-9  # float tolerance for 100.0

def as_float(x: Any) -> float:
    """Convert possible string/number to float."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        return float(x.strip())
    raise ValueError(f"Cannot convert {x!r} to float.")

def has_perfect(accs: Iterable[Any]) -> bool:
    """Return True if any accuracy is approximately 100.0 (0..100 scale)."""
    for a in accs:
        try:
            if as_float(a) >= 100.0 - EPS:
                return True
        except Exception:
            # Ignore non-convertible entries
            continue
    return False

def find_min_arch_per_canonical(data: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """
    For each canonical key, pick the FIRST architecture (in JSON order)
    that has at least one 100% accuracy element. If none, set empty string.
    """
    results = {}
    for canonical, arch_dict in data.items():
        min_arch = ""
        if isinstance(arch_dict, dict):
            for arch_name, acc_list in arch_dict.items():
                if isinstance(acc_list, (list, tuple)) and has_perfect(acc_list):
                    min_arch = arch_name
                    break
        results[canonical] = min_arch
    return results

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON must map canonical -> {architecture -> [accuracies]}")

    results = find_min_arch_per_canonical(data)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["canonical", "minimal_architecture"])
        for canonical, arch in results.items():
            w.writerow([canonical, arch])

    print(f"Wrote {OUTPUT_CSV} with {len(results)} rows.")

if __name__ == "__main__":
    main()

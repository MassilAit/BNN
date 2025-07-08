import json
import csv
from ast import literal_eval
from pathlib import Path

def find_min_arch(accuracies_map):
    """
    Given a dict mapping architecture keys (string repr. of list) to accuracy lists,
    return the first (simplest) architecture that has at least one 100.0 accuracy.
    Simplicity is determined by:
      1) length of the parsed list (fewer layers/units is simpler)
      2) lex order of the list as a tiebreaker
    """
    # Parse architecture keys into actual lists
    archs = [(literal_eval(key), key) for key in accuracies_map.keys()]
    # Sort by (length, lexicographically)
    archs.sort(key=lambda x: (len(x[0]), x[0]))
    # Find first architecture with any 100.0 accuracy
    for parsed, key in archs:
        if any(acc == 100.0 for acc in accuracies_map[key]):
            return key
    # If none achieve 100%, return an empty string or placeholder
    return ""

def json_to_csv(src_json: Path, dst_csv: Path):
    """
    Reads the JSON file at src_json and writes a CSV to dst_csv
    with columns: int_tt, minimal_architecture
    """
    src = Path(src_json)
    data = json.loads(src.read_text())
    dst = Path(dst_csv)
    
    with dst.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['int_tt', 'minimal_architecture'])
        # Sort rows by integer value of int_tt
        for int_tt in sorted(data.keys(), key=int):
            acc_map = data[int_tt]
            minimal = find_min_arch(acc_map)
            writer.writerow([int_tt, minimal])

# Example usage:
n=4
json_to_csv(f"Result/{n}_inputs/summary{n}.json", f"Result/{n}_inputs/minimal_architectures_{n}.csv")


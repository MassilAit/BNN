# classify_ternary.py
# -------------------------------------------------
# Assumes the DB file was created by ternary_classes.py
# and lives in   DB/ternary_class_db_n{n}.pkl.gz

import gzip
import pickle
from typing import Tuple

# --- 1.  load the DB --------------------------------------------------
def load_db(n: int, path_template: str = "DB/dc/ternary_class_db_n{}.pkl.gz"):
    with gzip.open(path_template.format(n), "rb") as f:
        return pickle.load(f)

# --- 2.  same permutation routine as in the builder ------------------
def apply_perm_key(key: int, row_map) -> int:
    care, value = key >> 16, key & 0xFFFF
    new_care = new_value = 0
    for old, new in enumerate(row_map):
        if (care >> old) & 1:
            new_care  |= 1 << new
            if (value >> old) & 1:
                new_value |= 1 << new
    return (new_care << 16) | new_value

# --- 3.  classify one function ---------------------------------------
def classify_function(care_mask: int,
                      value_mask: int,
                      db) -> Tuple[int, int]:
    """
    Returns (class_index, class_size) for the given ternary function.
    """
    key = (care_mask << 16) | value_mask
    canonical = min(apply_perm_key(key, rm) for rm in db["row_maps"])

    idx = db["rep2idx"][canonical]
    return idx, db["sizes"][idx]


# ---------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    db = load_db(4)                       # n = 4 database

    print(len(db["reps"]))
    
    ## ---- define your truth-table here --------------------------------
    ## rows with defined outputs (care = 1) – e.g. first 8 rows:
    #care  = int("0000_0000_0000_1011".replace("_", ""), 2)
#
    ## rows where the output is 1 (and care = 1) – just an example:
    #value = int("0000_0000_0000_1011".replace("_", ""), 2)
#
    #idx, size = classify_function(care, value, db)
    #print(f"Class index : {idx}")
    #print(f"Class size  : {size}")

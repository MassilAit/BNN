# ternary_classes.py  (with progress bar)
# -------------------------------------------------
# Enumerate 0/1/X truth tables for n inputs, group them
# by input permutations, and store one representative
# per equivalence class plus its size.
#
# Progress bar shows how far we are through the full
# 3^(2^n) enumeration.

import gzip
import pickle
from itertools import permutations
from typing import List, Dict

try:
    from tqdm import tqdm
except ImportError:                           # graceful fallback
    def tqdm(iterable, *a, **k):              # noqa: D401
        return iterable                       # just return the iterable


# ------------------------------------------------------------------
# STEP 1 – permutation maps
# ------------------------------------------------------------------
def compute_permutation_maps(n: int) -> List[List[int]]:
    """Return list of row-maps; one per permutation of input bits."""
    num_rows = 1 << n
    maps: List[List[int]] = []
    for perm in permutations(range(n)):
        row_map = [0] * num_rows
        for old in range(num_rows):
            new = 0
            for i in range(n):
                bit = (old >> i) & 1
                new |= bit << perm[i]
            row_map[old] = new
        maps.append(row_map)
    return maps


# ------------------------------------------------------------------
# STEP 2 – apply permutation to packed key
# ------------------------------------------------------------------
def apply_perm_key(key: int, row_map: List[int]) -> int:
    """Return packed key after permuting input bits."""
    care, value = key >> 16, key & 0xFFFF
    new_care = new_value = 0
    for old, new in enumerate(row_map):
        if (care >> old) & 1:
            new_care  |= 1 << new
            if (value >> old) & 1:
                new_value |= 1 << new
    return (new_care << 16) | new_value


# ------------------------------------------------------------------
# STEP 3 – iterate ALL ternary tables
# ------------------------------------------------------------------
def all_ternary_keys(n: int):
    """Yield every packed (care,value) key for 2**n rows."""
    rows = 1 << n
    for care in range(1 << rows):
        sub = care                      # iterate all subsets of 'care'
        while True:
            yield (care << 16) | sub    # sub == value mask
            if sub == 0:
                break
            sub = (sub - 1) & care      # next subset


# ------------------------------------------------------------------
# STEP 4 – build classes with progress bar
# ------------------------------------------------------------------
def permutation_classes_ternary(n: int):
    """Return (reps, sizes, row_maps)."""
    row_maps = compute_permutation_maps(n)
    seen:  set[int] = set()
    reps:  List[int] = []
    sizes: List[int] = []

    total_tables = pow(3, 1 << n)
    iterator = tqdm(all_ternary_keys(n),
                    total=total_tables,
                    unit="table",
                    smoothing=0.05,
                    colour="cyan",
                    desc=f"Enumerating n={n}")

    for key in iterator:
        if key in seen:
            continue

        class_size = 0
        min_rep    = key
        stack      = [key]

        while stack:
            f = stack.pop()
            if f in seen:
                continue
            seen.add(f)
            class_size += 1
            if f < min_rep:
                min_rep = f
            for rm in row_maps:
                g = apply_perm_key(f, rm)
                if g not in seen:
                    stack.append(g)

        reps.append(min_rep)
        sizes.append(class_size)

    iterator.close()
    return reps, sizes, row_maps


# ------------------------------------------------------------------
# STEP 5 – build DB and save
# ------------------------------------------------------------------
def build_db(n: int, out_path: str = "DB/dc/ternary_class_db_n{}.pkl.gz"):
    print(f"Building ternary class DB for n={n} ...")
    reps, sizes, row_maps = permutation_classes_ternary(n)

    db: Dict[str, object] = {
        "n": n,
        "row_maps": row_maps,
        "reps": reps,            # canonical representatives
        "sizes": sizes,          # class sizes
        "rep2idx": {rep: i for i, rep in enumerate(reps)},
    }

    out_file = out_path.format(n)
    with gzip.open(out_file, "wb") as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  → {len(reps)} classes found.")
    print(f"  ✅ saved to {out_file}")


# ------------------------------------------------------------------
# Example run
# ------------------------------------------------------------------
if __name__ == "__main__":
    # n = 4 ⇒ 3^16 = 43 046 721 tables
    # Run under PyPy or compile with Cython for speed.
    build_db(4)

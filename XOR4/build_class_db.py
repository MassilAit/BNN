import json
import numpy as np
import gzip
import pickle
from itertools import permutations, product
from scipy.optimize import linprog


# ==============================
# --- STEP 1: CLASS GENERATION
# ==============================

def compute_permutation_maps(n):
    num_rows = 1 << n
    all_perms = list(permutations(range(n)))
    all_row_maps = []

    for perm in all_perms:
        row_map = []
        for old_index in range(num_rows):
            new_index = 0
            for i in range(n):
                bit = (old_index >> i) & 1
                new_index |= bit << perm[i]
            row_map.append(new_index)
        all_row_maps.append(row_map)

    return all_row_maps


def apply_perm(func_int: int, row_map: list[int]) -> int:
    new_int = 0
    for old_row, new_row in enumerate(row_map):
        if func_int >> old_row & 1:
            new_int |= 1 << new_row
    return new_int


def permutation_classes(n: int):
    m = 1 << n
    max_fun = 1 << m
    row_maps = compute_permutation_maps(n)
    seen = set()
    classes = []

    for f in range(max_fun):
        if f in seen:
            continue
        cur_class = set()
        for row_map in row_maps:
            g = apply_perm(f, row_map)
            if g not in seen:
                seen.add(g)
                cur_class.add(g)
        classes.append(cur_class)

    return classes


# ==============================
# --- STEP 2: SEPARABILITY TEST
# ==============================

def generate_inputs(n: int) -> np.ndarray:
    return np.array(list(product([-1, 1], repeat=n)), dtype=float)


def int_to_truth_table(n: int, func_int: int) -> np.ndarray:
    return np.array([(func_int >> k) & 1 for k in range(2 ** n)], dtype=int)


def is_separable(outputs: np.ndarray, X_with_bias: np.ndarray) -> bool:
    y = 2 * outputs - 1.0
    A_ub = -(y[:, None] * X_with_bias)
    b_ub = -np.ones_like(y)
    c = np.zeros(X_with_bias.shape[1])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * X_with_bias.shape[1], method="highs")
    return res.success


def class_separability(classes: list[set[int]], n: int) -> list[bool]:
    X = generate_inputs(n)
    X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])
    results = []
    for cls in classes:
        rep_func = min(cls)
        truth_table = int_to_truth_table(n, rep_func)
        results.append(is_separable(truth_table, X_with_bias))
    return results


# ==============================
# --- STEP 3: BUILD DB + SAVE
# ==============================

def build_db(n: int, output_file: str = "DB/logic_class_db_n{}.pkl.gz"):
    print(f"Building class DB for n = {n}...")
    classes = permutation_classes(n)
    print(f"  → {len(classes)} unique classes found.")
    
    linsep = class_separability(classes, n)
    print("  → Separability computed.")

    func2class = {}
    for idx, cls in enumerate(classes):
        for f in cls:
            func2class[f] = idx

    db = {
        "n": n,
        "classes": [sorted(list(cls)) for cls in classes],  # Optional but useful
        "separability": linsep,
        "func2class": func2class
    }

    with gzip.open(output_file.format(n), "wb") as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  ✅ Saved DB to {output_file.format(n)}")


# ==============================
# --- RUN
# ==============================

if __name__ == "__main__":
    build_db(2)
    build_db(3)
    build_db(4)

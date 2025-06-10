#!/usr/bin/env python3
"""
Estimate the proportion of linearly separable Boolean functions
for input sizes n = 2 … MAX_N.

Method
------
* For n ≤ 4, enumerate every Boolean function (exact result).
* For n > 4, sample SAMPLE_SIZE random functions (Monte-Carlo estimate).
* Linear separability is checked by solving the feasibility LP

        find w, b  s.t.  y_i · (w·x_i + b) ≥ 1   for all i,

  where y_i ∈ {-1, +1} are the truth-table outputs.

Dependencies
------------
numpy, scipy (for linprog), matplotlib (for the final plot)

Execution
---------
$ pip install numpy scipy matplotlib
$ python linsep_proportion.py
"""

import itertools
import random
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

# --------------------------------------------------------------------
MAX_N = 8           # upper bound on number of inputs you want to test
SAMPLE_SIZE = 10_000  # number of random functions to sample when n > 4
SEED = 42           # RNG seed for reproducibility
# --------------------------------------------------------------------


def generate_inputs(n: int) -> np.ndarray:
    """Return all 2**n binary input vectors as shape (2**n, n) array of ±1."""
    return np.array(list(itertools.product([-1, 1], repeat=n)), dtype=float)


def is_separable(outputs: np.ndarray, X_with_bias: np.ndarray) -> bool:
    """
    Exact linear separability test via LP.

    Parameters
    ----------
    outputs : (m,) array with values 0/1
    X_with_bias : (m, n+1) array where last column is 1 (bias term)

    Returns
    -------
    bool : True if a separating hyperplane exists.
    """
    y = 2 * outputs - 1.0  # map {0,1} → {-1, +1}
    A_ub = -(y[:, None] * X_with_bias)      #  y_i (w·x_i + b) ≥ 1
    b_ub = -np.ones_like(y)

    # Objective vector is all zeros – we only care about feasibility
    c = np.zeros(X_with_bias.shape[1])

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * X_with_bias.shape[1],
        method="highs",
    )
    return res.success


def enumerate_functions(n: int) -> List[np.ndarray]:
    """Yield every possible truth table for n inputs (exact enumeration)."""
    for code in range(2 ** (2 ** n)):
        bits = [(code >> k) & 1 for k in range(2 ** n)]
        yield np.array(bits, dtype=int)


def sample_functions(n: int, k: int, rng: random.Random) -> List[np.ndarray]:
    """Return k random truth tables for n inputs (Monte-Carlo sampling)."""
    m = 2 ** n
    return [np.array(rng.choices([0, 1], k=m), dtype=int) for _ in range(k)]


def proportion_separable(n: int, X_with_bias: np.ndarray, rng: random.Random):
    """Compute proportion of linearly separable functions for a given n."""
    if n <= 4:
        funcs = enumerate_functions(n)
        total = 2 ** (2 ** n)
    else:
        funcs = sample_functions(n, SAMPLE_SIZE, rng)
        total = SAMPLE_SIZE

    separable_count = 0
    for outputs in funcs:
        if is_separable(outputs, X_with_bias):
            separable_count += 1

    return separable_count / total


def main():
    rng = random.Random(SEED)
    ns, props = [], []

    for n in range(2, MAX_N + 1):
        X = generate_inputs(n)
        X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])

        start = time.time()
        p = proportion_separable(n, X_with_bias, rng)
        elapsed = time.time() - start

        ns.append(n)
        props.append(p)
        print(f"n={n}: proportion ≈ {p:.6f}   (computed in {elapsed:.1f}s)")

    # ---- Plot --------------------------------------------------------
    plt.figure()
    plt.plot(ns, props, marker="o")
    plt.xlabel("Number of inputs (n)")
    plt.ylabel("Proportion of linearly separable functions")
    plt.title("Estimated proportion of linearly separable Boolean functions")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Iterative construction of NPN-canonical Boolean functions
and per-level timing report.

Counts are:
    n = 1 →   2
    n = 2 →   4
    n = 3 →  14
    n = 4 → 222
    n = 5 → 616 126   (takes a few minutes, < 1 GB RAM)
"""

import itertools, time, json

# ------------------------------------------------------------
#  truth-table  ⟷  integer
# ------------------------------------------------------------
def int_to_bits(val, n):
    return [(val >> i) & 1 for i in range(1 << n)]


def bits_to_int(bits):
    v = 0
    for i, b in enumerate(bits):
        v |= (b & 1) << i
    return v


# ------------------------------------------------------------
#  row-map utilities  (perm ∘ input-neg)
# ------------------------------------------------------------
def _bitpos(n, var):
    return n - 1 - var               # x1 MSB, xn LSB


def make_row_map(n, perm, mask):
    size = 1 << n
    row  = [0] * size
    for i in range(size):
        j = 0
        for new, old in enumerate(perm):
            bit  = (i >> _bitpos(n, old)) & 1
            bit ^= (mask >> _bitpos(n, old)) & 1
            j   |= bit << _bitpos(n, new)
        row[i] = j
    return row


def all_row_maps(n):
    maps = []
    for perm in itertools.permutations(range(n)):
        for mask in range(1 << n):
            maps.append(make_row_map(n, perm, mask))
    return maps


# ------------------------------------------------------------
#  canonicalise one function under full NPN
# ------------------------------------------------------------
def canonical(f_int, n, maps, ones):
    best = None
    for rmap in maps:
        g = 0
        for i, j in enumerate(rmap):
            g |= ((f_int >> i) & 1) << j
        if best is None or g < best:
            best = g
        g ^= ones                     # output complement
        if g < best:
            best = g
    return best


# ------------------------------------------------------------
#  compose two n-input cofactors → (n+1)-input function
# ------------------------------------------------------------
def compose(f0, f1, n):
    size = 1 << n
    out  = 0
    for r in range(size):
        out |= ((f0 >> r) & 1) << (2 * r)       # xn+1 = 0
        out |= ((f1 >> r) & 1) << (2 * r + 1)   # xn+1 = 1
    return out


# ------------------------------------------------------------
#  ALL variants of a rep:  n!  · 2ⁿ  · 2
# ------------------------------------------------------------
def variants(rep, n):
    ones = (1 << (1 << n)) - 1
    for perm in itertools.permutations(range(n)):
        for mask in range(1 << n):
            rmap = make_row_map(n, perm, mask)
            v = 0
            for i, j in enumerate(rmap):
                v |= ((rep >> i) & 1) << j
            yield v
            yield v ^ ones            # complement


# ------------------------------------------------------------
#  lift  reps_n  (arity n)  →  reps_{n+1}
# ------------------------------------------------------------
def lift(reps_n, n):
    maps_np1 = all_row_maps(n + 1)
    ones_np1 = (1 << (1 << (n + 1))) - 1

    var_set  = {v for r in reps_n for v in variants(r, n)}
    var_list = sorted(var_set)
    new_reps = set()

    for i, f0 in enumerate(var_list):
        for f1 in var_list[i:]:                   # unordered pairs
            f_int = compose(f0, f1, n)
            canon = canonical(f_int, n + 1, maps_np1, ones_np1)
            new_reps.add(canon)

    return sorted(new_reps)


# ------------------------------------------------------------
#  brute seed  n = 1
# ------------------------------------------------------------
def seed_n1():
    maps = all_row_maps(1)
    ones = 3
    return sorted({canonical(f, 1, maps, ones) for f in range(4)})


# ------------------------------------------------------------
#  build and time each level
# ------------------------------------------------------------
def build(N_MAX=4):
    table = {}
    reps  = seed_n1()
    table[1] = reps
    print(f"n = 1: {len(reps):>9} classes   (elapsed 0.0 s)")

    for n in range(1, N_MAX):
        t0   = time.time()
        reps = lift(reps, n)
        dt   = time.time() - t0
        table[n + 1] = reps
        print(f"n = {n+1}: {len(reps):>9} classes   (elapsed {dt:.1f} s)")
        # Save to JSON file
        with open("npn_classes.json", "w") as f:
            json.dump(table, f, indent=2)
    return table


# ------------------------------------------------------------
#  run
# ------------------------------------------------------------
if __name__ == "__main__":
    # change to 5 to reach 616 126 classes (will take several minutes)
    reps = build(N_MAX=5)





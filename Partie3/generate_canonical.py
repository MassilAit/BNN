import itertools

# ----------  Encoding helpers (Step 1) ----------
def tt_bits_to_int(bits):
    val = 0
    for i, b in enumerate(bits):
        val |= (b & 1) << i
    return val


def tt_int_to_bits(value, n):
    size = 1 << n
    return [(value >> i) & 1 for i in range(size)]


# ----------  Row-reindexing maps (Step 2) ----------
def _var_bitpos(n, var_idx):
    # x1 is MSB, xn is LSB
    return n - 1 - var_idx


def make_perm_neg_map(n, perm, mask):
    """row_map[i] = new index of row i after permuting and flipping inputs"""
    size = 1 << n
    out = [0] * size
    for i in range(size):
        j = 0
        for new_pos, old_pos in enumerate(perm):
            old_bit = (i >> _var_bitpos(n, old_pos)) & 1
            flip    = (mask >> _var_bitpos(n, old_pos)) & 1
            bit     = old_bit ^ flip
            j |= bit << _var_bitpos(n, new_pos)
        out[i] = j
    return out


def all_row_maps(n, perms=True, flips=True):
    """Cartesian product: every permutation × every input-flip mask"""
    perms_it = (itertools.permutations(range(n))
                if perms else [tuple(range(n))])
    masks_it = (range(1 << n) if flips else [0])
    for perm in perms_it:
        for mask in masks_it:
            yield make_perm_neg_map(n, perm, mask)




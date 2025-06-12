from generate_canonical import all_row_maps
import json


# ----------  Step 3: canonical form & orbit ----------
def canonical_and_orbit(f_int, row_maps, all_ones):
    best   = None
    orbit=set()
    for row_map in row_maps:
        g = 0
        # ---------------------------------------------------------------
        # row_map[i] = new index j ;  so copy bit(i) → position(j)
        # ---------------------------------------------------------------
        for i, j in enumerate(row_map):
            bit = (f_int >> i) & 1        # <-- OLD table row i
            g  |= bit << j                # --> NEW table row j
        # candidate 1
        orbit.add(g)
        if best is None or g < best:
            best = g
        # candidate 2 : output-negated
        g_flip = g ^ all_ones
        orbit.add(g_flip)
        if g_flip < best:
            best = g_flip

    return best,orbit



# ----------  Top-level enumeration for n ≤ 4 ----------
def enumerate_npn_classes(n):
    size_tt = 1 << (1 << n)         # 2^(2^n) truth tables
    all_ones = (1 << (1 << n)) - 1
    maps = list(all_row_maps(n))    # still tiny for n ≤ 4

    classes = {}                    # canon_int → [members...]

    for f in range(size_tt):
        canon, orbit = canonical_and_orbit(f, maps, all_ones)
        classes[canon]=list(orbit) 

    return classes

table={}
# ----------  quick demo ----------
if __name__ == "__main__":
        n=2
        print(enumerate_npn_classes(2))




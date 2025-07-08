import json
from pathlib import Path
from simpliqm import minimize, format_minimized_expression

# ---------- bit-helpers ----------
def int_to_minterms(tt_int: int, n_bits: int):
    """Return list of row indices where f == 1."""
    return [i for i in range(1 << n_bits) if (tt_int >> i) & 1]

def literal_count(implicant_str):
    """Count literals in a single implicant string like 'x1 x2' or '~x3'."""
    return sum(1 for tok in implicant_str if tok in {'0','1'})

def cost(implicants):
    """(terms, literals)  ->  tuple so you can pick your own metric."""
    terms = len(implicants)
    lits  = sum(literal_count(str(imp)) for imp in implicants)
    return terms, lits

# ------------------------------------------------------------------
#  Function-1 : complement each implicant (bin-string)                
#               SOP(f′)  ->  list[str] representing POS(f) terms   
# ------------------------------------------------------------------
def complement_implicants(terms):
    """
    Flip each bit in every Term:
        '0' -> '1'   (A'   becomes   A  inside a sum)
        '1' -> '0'   (A    becomes   A')
        '-' unchanged   (variable absent)
    Returns a list of plain strings so the existing cost() still works.
    """
    flipped = []
    for t in terms:                     # t is a simpliqm.Term
        s = str(t)
        flipped.append(''.join(
            '-' if c == '-' else ('0' if c == '1' else '1')
            for c in s
        ))
    return flipped                       # list[str]


# ------------------------------------------------------------------
#  Function-2 : pretty-print a POS expression from those bin-strings
# ------------------------------------------------------------------
def format_pos(bin_terms, names=None):
    """
    bin_terms : list of strings like '1-0'
    names     : optional list of variable names; default = ['A','B',…]
    """
    if not bin_terms:
        return "0"

    n_bits = len(bin_terms[0])
    if names is None:
        names = [chr(ord('A') + i) for i in range(n_bits)]  # A,B,C,…

    clauses = []
    for bits in bin_terms:
        lits = []
        for idx, bit in enumerate(bits):
            if bit == '-':
                continue
            var = names[idx]
            # bit was flipped already → 0 means un-complemented literal
            lits.append(var if bit == '1' else f"{var}'")
        clauses.append("(" + " + ".join(lits) + ")")
    return "".join(clauses)


def minimise_one(tt_int: int, n_bits: int, dont_cares=None):
    dont_cares = dont_cares or []

    ones  = int_to_minterms(tt_int, n_bits)
    
    zeros = [i for i in range(1 << n_bits)
             if i not in ones and i not in dont_cares]

    # exact SOP(f)
    sop = minimize(n_bits, ones,  dont_cares)
    sop_cost = cost(sop)

    sop_f_complement = minimize(n_bits, zeros, dont_cares)
    pos = complement_implicants(sop_f_complement) 
    pos_cost = cost(pos) 


    better = 'SOP' if sop_cost < pos_cost else 'POS'

    return {
        "SOP_terms" : format_minimized_expression(sop),
        "POS_terms" : format_pos(pos), #function2 should convert implicants with + and terms with ()
        "SOP_cost"  : sop_cost,   # (terms, literals)
        "POS_cost"  : pos_cost,
        "preferred" : better
    }


def minimise_one_ones(ones: list[int], n_bits: int, dont_cares=None):
    dont_cares = dont_cares or []

    zeros = [i for i in range(1 << n_bits)
             if i not in ones and i not in dont_cares]

    # exact SOP(f)
    sop = minimize(n_bits, ones,  dont_cares)
    sop_cost = cost(sop)

    sop_f_complement = minimize(n_bits, zeros, dont_cares)
    pos = complement_implicants(sop_f_complement) 
    pos_cost = cost(pos) 

    better = 'SOP' if sop_cost < pos_cost else 'POS'


    if better =='SOP':
        return format_minimized_expression(sop)
    else:
        return format_pos(pos)


def minimise_json_file(src_path: str | Path,
                       dst_path: str | Path,
                       dont_cares_map: dict[int, list[int]] | None = None):
    """
    src_path          : file with format
                          { "4": [ 11, 27, ... ],
                            "5": [ 123, ... ], ... }
    dst_path          : where to write the new JSON
    dont_cares_map    : optional { n_bits: [indices] }
                         — same mask reused for every TT of that size
                         — omit or pass {} if you have none
    """
    dont_cares_map = dont_cares_map or {}

    src_path = Path(src_path)
    dst_path = Path(dst_path)

    data_in  = json.loads(src_path.read_text())
    data_out = {}

    for n_str, tt_list in data_in.items():
        n_bits = int(n_str)
        dc     = dont_cares_map.get(n_bits, [])
        data_out[n_str] = {}

        for tt_int in tt_list:
            res = minimise_one(tt_int, n_bits, dc)
            data_out[n_str][str(tt_int)] = res

    dst_path.write_text(json.dumps(data_out, indent=2))
    print(f"Results written to {dst_path}")


if __name__ == "__main__":
    minimise_json_file("npn_classes_brute.json", "npn_classes_bool.json")



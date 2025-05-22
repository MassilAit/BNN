# ---------------------------------------------------------------
#  bnn_truth_tables.py   (pure-Python, no dependencies)
# ---------------------------------------------------------------
def sign(x: int) -> int:
    """Binary sign with sign(0)=+1  (as in your BinarySign)"""
    return 1 if x >= 0 else -1


INPUTS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # evaluation order


def decode_bits(n: int) -> tuple[int, ...]:
    """
    Map the integer 0-63 to the six weights (v2,v1,w22,w21,w12,w11),
    returning ±1 for each.
    """
    return tuple(
        -1 if ((n >> k) & 1) == 0 else 1
        for k in range(5, -1, -1)        # MSB→LSB   5 … 0
    )


def forward(weights: tuple[int, ...], x1: int, x2: int) -> int:
    """One forward pass through the 2-2-1 BNN (no bias)."""
    v2, v1, w22, w21, w12, w11 = weights

    z1 = sign(w11 * x1 + w12 * x2)
    z2 = sign(w21 * x1 + w22 * x2)
    y  = sign(v1 * z1 + v2 * z2)
    return y



network_csv  = ["bits,truth_table\n"]
function_map = {}     # truth_table  -> list[bits]

for n in range(64):
    bits_str = format(n, "06b")             # e.g. "010011"
    weights  = decode_bits(n)

    outputs = [forward(weights, *x) for x in INPUTS]
    tt = f'"({outputs[0]},{outputs[1]},{outputs[2]},{outputs[3]})"'

    network_csv.append(f"{bits_str},{tt}\n")
    function_map.setdefault(tt, []).append(bits_str)

# --- write CSV #1 --------------------------------------------------------
with open("network_truth_tables.csv", "w", encoding="utf-8") as f:
    f.writelines(network_csv)

# --- write CSV #2 --------------------------------------------------------
with open("function_networks.csv", "w", encoding="utf-8") as f:
    f.write("truth_table,count,networks_bits\n")
    for tt, bit_list in function_map.items():
        f.write(f'{tt},{len(bit_list)},"{" ".join(bit_list)}"\n')  # tt already quoted

print("✓ CSVs created: network_truth_tables.csv  &  function_networks.csv")




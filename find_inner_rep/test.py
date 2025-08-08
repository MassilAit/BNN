"""
Verify that the expressions stored in `SOP/POS` and `inner_representation`
columns reproduce the integer in `canonical`.

• Hard-codes the CSV path – edit CSV_FILE below if needed.
• Prints 'OK' when everything matches, else lists mismatches.
"""

CSV_FILE = "Result/4_continuous.csv"          # ← change if your file lives elsewhere
# ---------------------------------------------------------------------
from itertools import product
import pandas as pd

# ────────────────────────────── PARSER ────────────────────────────────
def _parse(expr: str):
    expr = expr.replace(" ", "")
    i = 0

    def peek(): return expr[i] if i < len(expr) else None
    def take(expected=None):
        nonlocal i
        if expected and peek() != expected:
            raise ValueError(f"expected '{expected}' at {i}")
        ch, i = expr[i], i + 1
        return ch

    def prim():
        ch = peek()
        if ch is None:
            raise ValueError("unexpected end")
        if ch.isalpha():
            return ("var", take())
        if ch in "([":                       # group
            close = ")" if ch == "(" else "]"
            take()
            inside = summation()
            if peek() != close:
                raise ValueError(f"missing '{close}' at {i}")
            take(close)
            return inside
        raise ValueError(f"unexpected '{ch}' at {i}")

    def factor():
        node = prim()
        while peek() == "'":
            take("'")
            node = ("not", node)
        return node

    def product_():
        nodes = [factor()]
        while True:
            ch = peek()
            if ch is None or ch in "+)]":
                break
            nodes.append(factor())
        return nodes[0] if len(nodes) == 1 else ("and", nodes)

    def summation():
        node = product_()
        while peek() == "+":
            take("+")
            node = ("or", node, product_())
        return node

    tree = summation()
    if i != len(expr):
        raise ValueError(f"trailing characters: {expr[i:]}")   # noqa: TRY003
    
    return tree


def _collect_vars(ast, acc=None):
    if acc is None:
        acc = set()
    typ = ast[0]
    if typ == "var":
        acc.add(ast[1])
    elif typ == "not":
        _collect_vars(ast[1], acc)
    elif typ == "and":
        for t in ast[1]:
            _collect_vars(t, acc)
    elif typ == "or":
        _collect_vars(ast[1], acc)
        _collect_vars(ast[2], acc)
    return acc


def _eval(ast, env):
    typ = ast[0]
    if typ == "var":
        return env[ast[1]]
    if typ == "not":
        return not _eval(ast[1], env)
    if typ == "and":
        return all(_eval(t, env) for t in ast[1])
    if typ == "or":
        return _eval(ast[1], env) or _eval(ast[2], env)
    raise RuntimeError(f"unknown node type {typ}")


# ─────────────────────────── PUBLIC API ──────────────────────────────
def truth_table(expr: str, n_inputs: int | None = None):
    ast = _parse(expr)
    all_names = sorted(_collect_vars(ast))

    if n_inputs is not None and n_inputs < len(all_names):
        raise ValueError(
            f"n_inputs={n_inputs} but expression uses "
            f"{len(all_names)} vars: {', '.join(all_names)}"
        )

    width = n_inputs if n_inputs is not None else len(all_names)

    table = {}
    for bits in product([0, 1], repeat=width):
        env = {v: bool(b) for v, b in zip(all_names, bits)}
        table[bits] = int(_eval(ast, env))
    return table


def bits_to_int(bits: list[int]) -> int:
    """Convert LSB-first list of 0/1 bits to integer."""
    return sum(bit << i for i, bit in enumerate(bits))

# ───────────────────────────── VERIFIER ──────────────────────────────
def check_csv(path: str) -> None:
    df = pd.read_csv(path)
    cols_to_check = [c for c in ("SOP/POS", "inner_representation") if c in df.columns]

    problems: list[tuple[int, str, str, int | None]] = []  # canonical, col, expr, got

    for _, row in df.iterrows():
        canonical = int(row["canonical"])
        n_input   = int(row["n_input"])

        for col in cols_to_check:
            expr = row[col]
            if not isinstance(expr, str) or not expr.strip():
                continue  # empty → ignore
            try:
                tt = truth_table(expr, n_inputs=n_input)
                outputs = [tt[b] for b in product([0, 1], repeat=n_input)]
                got = bits_to_int(outputs)
            except Exception as e:
                got  = None
                expr = f"{expr}  (ERROR: {e})"

            if got != canonical:
                problems.append((canonical, col, expr, got))

    if not problems:
        print("OK")
    else:
        print("MISMATCHES:")
        for can, col, expr, got in problems:
            print(f"  canonical {can} | column '{col}' | got {got}\n"
                  f"    expr: {expr}\n")

# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    check_csv(CSV_FILE)

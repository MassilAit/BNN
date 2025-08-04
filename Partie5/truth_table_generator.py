from itertools import product


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

    def product():
        nodes = [factor()]
        while True:
            ch = peek()
            if ch is None or ch in "+)]":
                break
            nodes.append(factor())
        return nodes[0] if len(nodes) == 1 else ("and", nodes)

    def summation():
        node = product()
        while peek() == "+":
            take("+")
            node = ("or", node, product())
        return node

    tree = summation()
    if i != len(expr):
        raise ValueError(f"trailing characters: {expr[i:]}")
    return tree


# ───────────────── VARIABLE GATHERING ────────────────────────────────
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


# ─────────────────────────── EVALUATOR ───────────────────────────────
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
    """
    Build a truth table for *expr*.
    If n_inputs is bigger than the variables used, extra inputs are ignored
    by the function but still appear in the key tuples.
    """
    ast = _parse(expr)
    all_names = sorted(_collect_vars(ast))        # real vars

    if n_inputs is not None and n_inputs < len(all_names):
        raise ValueError(
            f"n_inputs={n_inputs} but expression uses "
            f"{len(all_names)} vars: {', '.join(all_names)}"
        )

    width = n_inputs if n_inputs is not None else len(all_names)

    table = {}
    for bits in product([0, 1], repeat=width):
        env = {v: bool(b) for v, b in zip(all_names, bits)}   # map first bits
        table[bits] = int(_eval(ast, env))
    return all_names, table


# ───────────────────────────── DEMO ───────────────────────────────────
if __name__ == "__main__":
    expr = "A"
    vars_order, tt = truth_table(expr, n_inputs=2)
    print("Variables:", vars_order)
    for inp, out in tt.items():
        print(inp, "→", out)

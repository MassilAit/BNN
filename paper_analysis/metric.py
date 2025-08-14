from test import _parse


# ───────────────────────── Helpers for your AST ─────────────────────────
def _is(node, kind): return isinstance(node, tuple) and node and node[0] == kind
def _flatten_and(n):
    if not _is(n, 'and'): return [n]
    out = []
    for k in n[1]:
        out.extend(_flatten_and(k) if _is(k, 'and') else [k])
    return out

def _flatten_or(n):
    out = []
    def dfs(x):
        if _is(x, 'or'):
            _, a, b = x
            dfs(a); dfs(b)
        else:
            out.append(x)
    dfs(n)
    return out

def _make_or_chain(items):
    from functools import reduce
    def _or(a,b): return ('or', a, b)
    if not items: raise ValueError("empty OR chain")
    return reduce(_or, items)

# ───────────────────────── Push NOTs to leaves (NNF) ───────────────────
def nnf(n):
    if _is(n, 'var'): return n
    if _is(n, 'not'):
        x = n[1]
        if _is(x, 'var'): return n                     # literal already
        if _is(x, 'not'):  return nnf(x[1])            # ¬¬x => x
        if _is(x, 'and'):  # ¬(∧) => ∨ of negations
            kids = [_is(k,'and') and ('and', _flatten_and(k)) or k for k in x[1]]
            return _make_or_chain([nnf(('not', k)) for k in kids])
        if _is(x, 'or'):   # ¬(∨) => ∧ of negations
            _, a, b = x
            return ('and', [nnf(('not', a)), nnf(('not', b))])
        raise ValueError(f"bad node under not: {x}")
    if _is(n, 'and'):
        return ('and', [nnf(k) for k in _flatten_and(n)])
    if _is(n, 'or'):
        _, a, b = n
        return ('or', nnf(a), nnf(b))
    raise ValueError(f"unknown node: {n}")

# ─────────────────────── Tree (no sharing) metrics ──────────────────────
def literal_occurrences(n):
    if _is(n, 'var'): return 1
    if _is(n, 'not'): return literal_occurrences(n[1]) # don't double count
    if _is(n, 'and'): return sum(literal_occurrences(k) for k in _flatten_and(n))
    if _is(n, 'or'):
        _, a, b = n
        return literal_occurrences(a) + literal_occurrences(b)
    return 0

def gate_counts_tree(n):
    if _is(n, 'var'): return {'and':0,'or':0,'not':0}
    if _is(n, 'not'):
        c = gate_counts_tree(n[1]); c['not'] += 1; return c
    if _is(n, 'and'):
        kids = _flatten_and(n)
        total = {'and':1,'or':0,'not':0}
        for k in kids:
            sub = gate_counts_tree(k)
            for g in total: total[g] += sub[g]
        return total
    if _is(n, 'or'):
        _, a, b = n
        ca, cb = gate_counts_tree(a), gate_counts_tree(b)
        return {'and': ca['and']+cb['and'], 'or': ca['or']+cb['or']+1, 'not': ca['not']+cb['not']}
    raise ValueError(f"unknown node: {n}")

def two_input_eq_tree(n):
    if _is(n, 'var'): return 0
    if _is(n, 'not'): return 1 + two_input_eq_tree(n[1])  # count inverters as 1
    if _is(n, 'and'):
        kids = _flatten_and(n)
        return (len(kids)-1) + sum(two_input_eq_tree(k) for k in kids)
    if _is(n, 'or'):
        items = _flatten_or(n)
        return (len(items)-1) + sum(two_input_eq_tree(k) for k in items)
    raise ValueError

def depth(n, count_not=False):
    if _is(n, 'var'): return 0
    if _is(n, 'not'): return (1 if count_not else 0) + depth(n[1], count_not)
    if _is(n, 'and'):
        kids = _flatten_and(n)
        return 1 + (max(depth(k, count_not) for k in kids) if kids else 0)
    if _is(n, 'or'):
        items = _flatten_or(n)
        return 1 + max(depth(k, count_not) for k in items)
    raise ValueError

# ───────────────────── DAG (with sharing) metrics ───────────────────────
def _canon(n):
    if _is(n, 'var'): return ('v', n[1])
    if _is(n, 'not'):
        c = _canon(n[1]); return ('n', c)
    if _is(n, 'and'):
        keys = tuple(sorted(_canon(k) for k in _flatten_and(n)))
        return ('a', keys)
    if _is(n, 'or'):
        items = tuple(sorted(_canon(k) for k in _flatten_or(n)))
        return ('o', items)
    raise ValueError

def two_input_eq_dag(n):
    seen = set()
    def dfs(x):
        key = _canon(x)
        if key in seen: return 0
        seen.add(key)
        if _is(x, 'var'): return 0
        if _is(x, 'not'): return 1 + dfs(x[1])
        if _is(x, 'and'):
            kids = _flatten_and(x)
            return (len(kids)-1) + sum(dfs(k) for k in kids)
        if _is(x, 'or'):
            items = _flatten_or(x)
            return (len(items)-1) + sum(dfs(k) for k in items)
        raise ValueError
    return dfs(n)

def gate_counts_dag(n):
    seen = set()
    def dfs(x):
        key = _canon(x)
        if key in seen: return {'and':0,'or':0,'not':0}
        seen.add(key)
        if _is(x, 'var'): return {'and':0,'or':0,'not':0}
        if _is(x, 'not'):
            c = dfs(x[1]); c['not'] += 1; return c
        if _is(x, 'and'):
            tot = {'and':1,'or':0,'not':0}
            for k in _flatten_and(x):
                sub = dfs(k)
                for g in tot: tot[g] += sub[g]
            return tot
        if _is(x, 'or'):
            tot = {'and':0,'or':1,'not':0}
            for k in _flatten_or(x):
                sub = dfs(k)
                for g in tot: tot[g] += sub[g]
            return tot
        raise ValueError
    return dfs(n)

# ─────────────────────────── Top-level API ──────────────────────────────
def analyze_multilevel(tree):
    t = nnf(tree)  # push NOTs to leaves for fair counting
    stats_tree = {
        'two_input_eq': two_input_eq_tree(t),
        'gate_counts':  gate_counts_tree(t),
        'depth_no_not': depth(t, False),
        'depth_with_not': depth(t, True),
        'literal_occurrences': literal_occurrences(t),
    }
    stats_dag = {
        'two_input_eq': two_input_eq_dag(t),
        'gate_counts':  gate_counts_dag(t),
    }
    return {'tree': stats_tree, 'dag': stats_dag}

def _multilevel_cost(stats, prefer='dag'):
    s = stats['dag'] if prefer=='dag' else stats['tree']
    # Lexicographic: 2-input-eq → depth (no NOT) → literal occurrences
    return (s['two_input_eq'], stats['tree']['depth_no_not'], stats['tree']['literal_occurrences'])


def get_expression_stats(expr:str):
    return analyze_multilevel(_parse(expr))
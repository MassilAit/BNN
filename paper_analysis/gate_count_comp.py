# summarize_rich_plus.py
# Summarize a CSV of network vs SOP/POS stats.
# Prints per n_input:
#   - 5-bin counts (beats / same&same / same&bigger / worse 1–2 / worse >2)
#   - overall medians & means of Δcost (DAG 2-input eq) and Δdepth
#   - overall absolute gates (median & mean) for NN and QM (DAG 2-input eq)
#   - per-category medians (Δcost, Δdepth)
#   - top-5 biggest wins and losses (by (Δcost, Δdepth)), with canonical + costs

import pandas as pd
import json, ast
from collections import defaultdict
from statistics import mean, median

# ── SET YOUR CSV PATH HERE ─────────────────────────────────────────────
CSV_PATH = "Result/3_binarized.csv"
# ───────────────────────────────────────────────────────────────────────

def _parse_stats_cell(val):
    """Parse a stats cell (JSON string) into dict."""
    if pd.isna(val):
        return None
    s = str(val)
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None

def _extract_dag_cost_and_depth(stats):
    """
    Returns (dag_cost, depth_no_not) or (None, None).
    Expected:
      stats["cost"] = [tree, dag]
      stats["depth_no_not"] = int
    """
    try:
        dag = int(stats["cost"][1])
        depth = int(stats["depth_no_not"])
        return dag, depth
    except Exception:
        return None, None

def categorize(net_dag, net_depth, qm_dag, qm_depth):
    """
    Categories:
      - 'beats' (net_dag < qm_dag) OR (equal and net_depth < qm_depth)
      - 'same_gates_same_depth'
      - 'same_gates_bigger_depth'
      - 'slightly_worse_1_2' (net_dag > qm_dag and diff <= 2)
      - 'much_worse_gt2' (diff > 2)
      - 'skipped' if any missing
    """
    if None in (net_dag, net_depth, qm_dag, qm_depth):
        return 'skipped'
    if net_dag < qm_dag or (net_dag == qm_dag and net_depth < qm_depth):
        return 'beats'
    if net_dag == qm_dag and net_depth == qm_depth:
        return 'same_gates_same_depth'
    if net_dag == qm_dag and net_depth > qm_depth:
        return 'same_gates_bigger_depth'
    diff = net_dag - qm_dag
    if diff <= 2:
        return 'slightly_worse_1_2'
    return 'much_worse_gt2'

def summarize(csv_path: str):
    df = pd.read_csv(csv_path)
    if "network_stats" not in df.columns or "sop_pos_stats" not in df.columns:
        raise ValueError("CSV must contain 'network_stats' and 'sop_pos_stats' columns.")

    # Grouped structures per n_input
    counts = defaultdict(lambda: defaultdict(int))
    deltas = defaultdict(lambda: {"dcost": [], "ddepth": []})
    percat = defaultdict(lambda: defaultdict(lambda: {"dcost": [], "ddepth": []}))
    # absolute gate counts (DAG 2-input eq)
    abs_stats = defaultdict(lambda: {"nn_gates": [], "qm_gates": []})
    # rows for top-k lists
    rows_info = defaultdict(list)

    for _, row in df.iterrows():
        try:
            n = int(row["n_input"])
        except Exception:
            continue

        net_stats = _parse_stats_cell(row.get("network_stats"))
        qm_stats  = _parse_stats_cell(row.get("sop_pos_stats"))  # SOP/POS comparator

        net_dag, net_depth = _extract_dag_cost_and_depth(net_stats or {})
        qm_dag,  qm_depth  = _extract_dag_cost_and_depth(qm_stats or {})

        cat = categorize(net_dag, net_depth, qm_dag, qm_depth)
        counts[n][cat] += 1
        counts[n]['total'] += 1

        if cat == 'skipped':
            continue

        dcost  = net_dag - qm_dag
        ddepth = net_depth - qm_depth
        deltas[n]["dcost"].append(dcost)
        deltas[n]["ddepth"].append(ddepth)

        # absolute gates for both approaches
        abs_stats[n]["nn_gates"].append(net_dag)
        abs_stats[n]["qm_gates"].append(qm_dag)

        # per-category medians
        percat[n][cat]["dcost"].append(dcost)
        percat[n][cat]["ddepth"].append(ddepth)

        # keep row for top-k
        rows_info[n].append({
            "canonical": row.get("canonical"),
            "net_dag": net_dag, "qm_dag": qm_dag,
            "net_depth": net_depth, "qm_depth": qm_depth,
            "dcost": dcost, "ddepth": ddepth
        })

    # ── Print summaries per n_input ───────────────────────────────────
    for n in sorted(counts.keys()):
        c = counts[n]
        print(f"\n================  n_input = {n}  ================")
        print(f"Total functions:                {c.get('total',0)}")
        print(f"Beats McCluskey:                {c.get('beats',0)}")
        print(f"Same gates & same depth:        {c.get('same_gates_same_depth',0)}")
        print(f"Same gates, bigger depth:       {c.get('same_gates_bigger_depth',0)}")
        print(f"Worse by 1–2 gates:             {c.get('slightly_worse_1_2',0)}")
        print(f"Worse by >2 gates:              {c.get('much_worse_gt2',0)}")
        if c.get('skipped',0):
            print(f"Skipped (bad/missing stats):    {c.get('skipped',0)}")

        # Overall Δ stats
        if deltas[n]["dcost"]:
            print("\n-- Overall Δ stats (NN−QM) --")
            print(f"Median Δcost (DAG 2-input eq):  {median(deltas[n]['dcost']):.3f}")
            print(f"Mean   Δcost (DAG 2-input eq):  {mean(deltas[n]['dcost']):.3f}")
            print(f"Median Δdepth:                   {median(deltas[n]['ddepth']):.3f}")
            print(f"Mean   Δdepth:                   {mean(deltas[n]['ddepth']):.3f}")
        else:
            print("\n-- Overall Δ stats (NN−QM) --")
            print("No valid rows.")

        # Overall absolute gates (DAG 2-input eq)
        if abs_stats[n]["nn_gates"]:
            nn_med = median(abs_stats[n]["nn_gates"])
            nn_mean = mean(abs_stats[n]["nn_gates"])
            qm_med = median(abs_stats[n]["qm_gates"])
            qm_mean = mean(abs_stats[n]["qm_gates"])
            print("\n-- Overall absolute gates (DAG 2-input eq) --")
            print(f"NN gates: median={nn_med:.3f}, mean={nn_mean:.3f}")
            print(f"QM gates: median={qm_med:.3f}, mean={qm_mean:.3f}")
        else:
            print("\n-- Overall absolute gates (DAG 2-input eq) --")
            print("No valid rows.")

        # Per-category medians
        print("\n-- Per-category medians (Δcost, Δdepth) --")
        for cat_key in ["beats","same_gates_same_depth","same_gates_bigger_depth","slightly_worse_1_2","much_worse_gt2"]:
            vals = percat[n][cat_key]
            if vals["dcost"]:
                print(f"{cat_key:28s}  ({median(vals['dcost']):.3f}, {median(vals['ddepth']):.3f})   [n={len(vals['dcost'])}]")
            else:
                print(f"{cat_key:28s}  (na, na)   [n=0]")

        # Top-5 best and worst
        wins = sorted(rows_info[n], key=lambda r: (r["dcost"], r["ddepth"]))[:5]
        losses = sorted(rows_info[n], key=lambda r: (-r["dcost"], -r["ddepth"]))[:5]

        print("\n-- Top-5 biggest wins (most negative Δcost; tiebreak Δdepth) --")
        if wins:
            for r in wins:
                print(f"canonical={r['canonical']}  Δcost={r['dcost']:+d}  Δdepth={r['ddepth']:+d}  "
                      f"[NN_dag={r['net_dag']}, QM_dag={r['qm_dag']}; NN_depth={r['net_depth']}, QM_depth={r['qm_depth']}]")
        else:
            print("None")

        print("\n-- Top-5 biggest losses (most positive Δcost; tiebreak Δdepth) --")
        if losses:
            for r in losses:
                print(f"canonical={r['canonical']}  Δcost={r['dcost']:+d}  Δdepth={r['ddepth']:+d}  "
                      f"[NN_dag={r['net_dag']}, QM_dag={r['qm_dag']}; NN_depth={r['net_depth']}, QM_depth={r['qm_depth']}]")
        else:
            print("None")

if __name__ == "__main__":
    summarize(CSV_PATH)

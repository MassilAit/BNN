from generate_minimum_network import find_minimal_configuration
import os
from multiprocessing import Process, cpu_count
import json
import pandas as pd



def analyze_sublist(n_input: int, canonical_values: list, index: int):
    out_path = f"Result/{n_input}_inputs/summary_part_{index}.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    header_written = os.path.exists(out_path)

    for value in canonical_values:
        print(f"[{index}] Testing n={n_input}, function={value}")
        result = find_minimal_configuration(n_input, value)

        row = {
            "Canonical": value,
            "No_hidden": "",
            "Single_hidden": "",
            "Multiple": ""
        }

        if result == []:
            row["No_hidden"] = "x"
        else:
            for entry in result:
                if entry[0] == "single":
                    row["Single_hidden"] = str(entry[1])
                elif entry[0] == "multi":
                    row["Multiple"] = str(entry[1])

        df = pd.DataFrame([row])  # one-row DataFrame
        df.to_csv(out_path, index=False, mode='a', header=not header_written)
        header_written = True

    print(f"[{index}] Done → {out_path}")


def split_list(lst, k):
    return [lst[i::k] for i in range(k)]


def analyze_all_canonical_forms_parallel(n_input: int, canonical_values: list, num_chunks: int = None):
    num_chunks = num_chunks or min(cpu_count(), 8)
    sublists = split_list(canonical_values, num_chunks)

    result_dir = f"Result/{n_input}_inputs"
    os.makedirs(result_dir, exist_ok=True)

    processes = []
    for i, sublist in enumerate(sublists):
        p = Process(target=analyze_sublist, args=(n_input, sublist, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge all summaries
    dfs = []
    for i in range(num_chunks):
        part_path = os.path.join(result_dir, f"summary_part_{i}.csv")
        dfs.append(pd.read_csv(part_path))
 

    final_df = pd.concat(dfs, ignore_index=True)

    # (1) sort rows by Canonical
    final_df.sort_values("Canonical", inplace=True)

    # (2) ensure Single_hidden is int, not float
    final_df["Single_hidden"] = (
        pd.to_numeric(final_df["Single_hidden"], errors="coerce")
          .astype("Int64")      # nullable integer; blanks stay blank
    )

    # (3) keep the intermediate files – **no os.remove(...)**

    final_df.to_csv(os.path.join(result_dir, "summary.csv"), index=False)
    print("Final merged summary saved to Result/{n_input}_inputs/summary.csv")


def analyze_npn_json(n_input:int, json_path: str = "npn_classes_brute.json"):
    """
    Loads NPN classification data from JSON and analyzes each input size group.

    Inputs:
        - json_path: Path to a JSON file mapping str(n_input) -> list of canonical values

    Output:
        - None (calls analyze_all_canonical_forms for each n_input)
    """
    with open(json_path, "r") as f:
        npn_classes = json.load(f)

   
    canonical_values = npn_classes[str(n_input)]

    print(f"\n=== Analyzing all functions for {n_input} inputs ===")
    analyze_all_canonical_forms_parallel(n_input, canonical_values)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  

    analyze_npn_json(4)



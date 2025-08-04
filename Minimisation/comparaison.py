import pandas as pd

def compare_csv_files(file1, file2):
    df1 = pd.read_csv(file1, dtype=str).fillna("")
    df2 = pd.read_csv(file2, dtype=str).fillna("")

    # Normalize spaces and strings
    df1 = df1.applymap(lambda x: str(x).strip())
    df2 = df2.applymap(lambda x: str(x).strip())

    # Compare row by row
    diff_mask = (df1 != df2)
    differing_rows = df1[diff_mask.any(axis=1)]

    # Optionally print which cells differ
    print("Rows with at least one different cell:")
    print(len(differing_rows))

    return differing_rows

def summarize_logic_results(file:str):

    df = pd.read_csv(file)
    # Count rows where "No_hidden" is 'x' (after stripping and string conversion)
    no_hidden_x_count = (df["No_hidden"].astype(str).str.strip() == "x").sum()

    # Find the maximum numeric value in "Single_hidden" (ignoring empty or non-numeric)
    def to_int_or_none(x):
        try:
            return int(x)
        except:
            return None

    single_hidden_cleaned = df["Single_hidden"].apply(to_int_or_none)
    max_single_hidden = single_hidden_cleaned.max()

    print(f"Number of rows with No_hidden == 'x': {no_hidden_x_count}")
    print(f"Maximum Single_hidden value: {max_single_hidden}")


# Example usage
#result = compare_csv_files("4_tanh_1.csv", "4_tanh_2.csv")
#result = compare_csv_files("4_sign_1.csv", "4_sign_2.csv")
#result = compare_csv_files("4_tanh_1.csv", "4_sign_2.csv")
#result.to_csv("differences.csv", index=False)


summarize_logic_results("4_tanh_2.csv")
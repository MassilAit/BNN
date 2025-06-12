import csv
import os
from model import BinarizedModel
from dataset import LogicDataSet
from train import train_model, train_model_activations
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import json

class BCEPlusMinusOne(nn.Module):
    """
    BCE-with-logits but accepts targets in {-1, +1}.
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets_pm1):
        targets01 = (targets_pm1 + 1) / 2       # map −1→0, +1→1
        return self.bce(logits, targets01)
    

def test_output(data, model):

    model.eval()
    for X,Y_TRUE in data:
        with torch.no_grad():
            X=X.unsqueeze(0)
            logit = model(X)
            Y_PRED = torch.sign(logit)[0][0]  # safely index scalar
            Y_TRUE = Y_TRUE[0] if Y_TRUE.dim() > 0 else Y_TRUE  # also make sure Y_TRUE is scalar

            if Y_PRED.item() == 0:
                Y_PRED = torch.tensor(1., dtype=Y_TRUE.dtype)

            if Y_PRED.item() != Y_TRUE.item():
                return False

    
    return True


def train_one(n_input, n_hidden, data:DataLoader, output:LogicDataSet, value, N_EPOCHS):
    model=BinarizedModel(n_input, n_hidden, bias = True)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = BCEPlusMinusOne()
    train_model(model,data,optimizer,N_EPOCHS,criterion)

    if test_output(output, model):
        print("Loading model")
        save_path = f"Result/{n_input}_inputs/{value}/{n_hidden}_weights.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        return True
    
    return False

def test_one(n_input, n_hidden, data:DataLoader, output:LogicDataSet, value, N_EPOCHS = 10000):
    print(f"Testing : {n_hidden}")
    for i in range(2):
        if train_one(n_input, n_hidden, data, output, value, N_EPOCHS):
            return True
        
    return False


def find_one(n_input, value, max_factor=3):
    data = LogicDataSet(n_input, value)
    batch_size = max((1 << n_input) // 2, 16)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)

    results = []

    # --- 1. No hidden layer ---
    if test_one(n_input, [], train_loader, data, value):
        return []  # If no hidden layer works, we're done

    # --- 2. Single hidden layer ---
    single_success = False
    for width in range(n_input, (1 << n_input) + 1):
        if test_one(n_input, [width], train_loader, data,  value):
            results.append(["single", width])
            single_success = True
            break  # stop at minimal working width

    # --- 3. Two hidden layers ---
    multi_success = False
    max_total_neurons = max_factor * n_input
    for total_neurons in range(n_input+1, max_total_neurons + 1):
        for h1 in range(n_input, total_neurons):
            h2 = total_neurons - h1
            if test_one(n_input, [h1, h2], train_loader, data, value):
                results.append(["multi", [h1, h2]])
                multi_success = True
                break
        if multi_success:
            break

    return results if results else ["fail", None]



def test_all_canonical(n, canonical_list):
    summary = []

    result_dir = f"Result/{n}_inputs"
    os.makedirs(result_dir, exist_ok=True)
    summary_path = os.path.join(result_dir, "summary.csv")

    for value in canonical_list:
        print(f"Testion n = {n}, value = {value}.")
        result = find_one(n, value)
        
        row = {
            "Canonical": value,
            "No_hidden": "",
            "Single_hidden": "",
            "Multiple": ""
        }

        if result == []:
            row["No_hidden"] = "X"
        else:
            for entry in result:
                if entry[0] == "single":
                    row["Single_hidden"] = str(entry[1])
                elif entry[0] == "multi":
                    row["Multiple"] = "-".join(str(x) for x in entry[1])

        summary.append(row)

        # Write to CSV
        with open(summary_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Canonical", "No_hidden", "Single_hidden", "Multiple"])
            writer.writeheader()
            writer.writerows(summary)

    print(f"Saved summary to {summary_path}")



def test_all_npn_classes(json_path="npn_classes.json"):
    with open(json_path, "r") as f:
        npn_data = json.load(f)

    for n_str, canonical_list in npn_data.items():
        n = int(n_str)
        print(f"\n=== Testing NPN classes with {n} inputs ===")
        test_all_canonical(n, canonical_list)



with open("npn_classes.json", "r") as f:
    npn_data = json.load(f)

n = 4
print(f"\n=== Testing NPN classes with {n} inputs ===")

test_all_canonical(n, npn_data[str(n)])



    


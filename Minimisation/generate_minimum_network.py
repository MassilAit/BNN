import csv
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import BinarizedModel
from dataset import LogicDataSet
from train import train_model


class BCEPlusMinusOne(nn.Module):
    """
    Custom loss: Binary Cross-Entropy with Logits, but for targets in {-1, +1}.

    Inputs:
        - logits: Tensor of raw model outputs (before activation), shape (batch_size, 1)
        - targets_pm1: Tensor of ground truth labels in {-1, +1}, shape (batch_size, 1)

    Output:
        - loss: Scalar tensor representing the loss value
    """
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets_pm1):
        targets_01 = (targets_pm1 + 1) / 2  # map −1→0, +1→1
        return self.bce(logits, targets_01)


def evaluate_model_accuracy(dataset: LogicDataSet, model: nn.Module) -> bool:
    """
    Checks if the model correctly predicts all outputs in a logic dataset.

    Inputs:
        - dataset: LogicDataSet, a PyTorch Dataset yielding (X, Y) with Y in {-1, +1}
        - model: Trained neural network model (inherits from nn.Module)

    Output:
        - True if all predictions are correct, False otherwise
    """
    model.eval()
    for X, Y_TRUE in dataset:
        with torch.no_grad():
            X = X.unsqueeze(0)
            logit = model(X)
            Y_PRED = torch.sign(logit)[0][0]
            Y_TRUE = Y_TRUE[0] if Y_TRUE.dim() > 0 else Y_TRUE

            if Y_PRED.item() == 0:
                Y_PRED = torch.tensor(1., dtype=Y_TRUE.dtype)

            if Y_PRED.item() != Y_TRUE.item():
                return False
    return True


def try_training_and_saving(n_input: int, hidden_layers: list, data_loader: DataLoader,
                            eval_dataset: LogicDataSet, value: str, epochs: int) -> bool:
    """
    Trains a model on a given dataset and saves it if it perfectly learns the function.

    Inputs:
        - n_input: Number of input bits to the logic function
        - hidden_layers: List of integers representing number of neurons per hidden layer
                         (e.g., [] for no hidden layer, [4], or [4,2])
        - data_loader: DataLoader to feed the training data
        - eval_dataset: Dataset used for evaluation (should contain all logic examples)
        - value: String representing the canonical logic function (truth table in binary)
        - epochs: Maximum number of training epochs

    Output:
        - True if model reaches 100% accuracy and is saved, False otherwise
    """

    model = BinarizedModel(n_input, hidden_layers, bias=True)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = BCEPlusMinusOne()
    train_model(model, data_loader, optimizer, epochs, criterion, early_stop=True)

    if evaluate_model_accuracy(eval_dataset, model):
        save_path = f"Result/{n_input}_inputs/{value}/{hidden_layers}_weights.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        return True

    return False


def try_multiple_trainings(n_input: int, hidden_layers: list, data_loader: DataLoader,
                           eval_dataset: LogicDataSet, value: str,
                           attempts: int = 10, epochs: int = 10000) -> bool:
    """
    Repeats model training multiple times for robustness against initialization.

    Inputs:
        - n_input: Number of input bits
        - hidden_layers: Network architecture as list of layer widths
        - data_loader: DataLoader for training
        - eval_dataset: Dataset for accuracy evaluation
        - value: Canonical logic function as a binary string
        - attempts: Number of retries
        - epochs: Training epochs per attempt

    Output:
        - True if any attempt succeeded (100% accuracy), False otherwise
    """
    print(f"Testing configuration: {hidden_layers}")
    for _ in range(attempts):
        if try_training_and_saving(n_input, hidden_layers, data_loader, eval_dataset, value, epochs):
            return True
    return False


def find_minimal_configuration(n_input: int, target_value: str, max_factor: int = 3) -> list:
    """
    Tries different model architectures (0, 1, or 2 hidden layers) to find the minimal 
    one that can learn the logic function.

    Inputs:
        - n_input: Number of logic inputs
        - target_value: String of bits representing the logic function's truth table
        - max_factor: Maximum total neurons allowed is (max_factor * n_input)

    Output:
        - [] if no hidden layer works
        - [["single", width]] if a single-layer model works
        - [["multi", [width1, width2]]] if two-layer model is needed
        - ["fail", None] if nothing works
    """

    dataset = LogicDataSet(n_input, target_value)
    batch_size = max((1 << n_input) // 2, 16)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Try without hidden layers
    if try_multiple_trainings(n_input, [], data_loader, dataset, target_value):
        return []

    results = []

    # Try single hidden layer
    for width in range(2, (1 << n_input) + 1):
        if try_multiple_trainings(n_input, [width], data_loader, dataset, target_value):
            results.append(["single", width])
            break

    # Try two hidden layers (bounded total neurons)
    for total in range(4, (1 << n_input) + 1):
        valid_splits=[]
        for h1 in range(2, total-1):
            h2 = total - h1
            if (h2<=h1):
                if try_multiple_trainings(n_input, [h1, h2], data_loader, dataset, target_value):
                    valid_splits.append([h1,h2])
    
        if valid_splits :
            results.append(["multi", valid_splits])
            return results
    
    return results if results else ["fail", None]


def analyze_all_canonical_forms(n_input: int, canonical_values: list):
    """
    Tests all canonical logic functions for a given number of inputs and records
    the minimum working architectures in a CSV file.

    Inputs:
        - n_input: Number of logic inputs
        - canonical_values: List of binary strings representing canonical truth tables

    Output:
        - None (writes result to Result/{n_input}_inputs/summary.csv)
    """
    summary = []
    result_dir = f"Result/{n_input}_inputs"
    os.makedirs(result_dir, exist_ok=True)
    summary_path = os.path.join(result_dir, "summary.csv")

    for value in canonical_values:
        print(f"\nTesting n={n_input}, function={value}")
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

        summary.append(row)

        # Save progress incrementally
        with open(summary_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(summary)

    print(f"Saved summary to {summary_path}")


def analyze_npn_json(json_path: str = "npn_classes_brute.json"):
    """
    Loads NPN classification data from JSON and analyzes each input size group.

    Inputs:
        - json_path: Path to a JSON file mapping str(n_input) -> list of canonical values

    Output:
        - None (calls analyze_all_canonical_forms for each n_input)
    """
    with open(json_path, "r") as f:
        npn_classes = json.load(f)

    for n_str, canonical_values in npn_classes.items():
        n_input = int(n_str)
        if n_input ==4 :
            break
        print(f"\n=== Analyzing all functions for {n_input} inputs ===")
        analyze_all_canonical_forms(n_input, canonical_values)


if __name__ == "__main__":
    analyze_npn_json()

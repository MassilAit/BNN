import torch
from torch.utils.data import Dataset

class LUTDataSet(Dataset):
    """
    A dataset that uses all LUT data based on a user-defined LUT (Look Up Table).

    Args:
        lut (dict): A LUT for the logic function used.

    Attributes:
        lut (dict): The LUT to generate outputs.
        n_data (int): Number of data samples.
    """
    def __init__(self, lut):
        self.lut = lut
        self.inputs = [input for input in lut.keys()]
        self.n_data = len(lut)

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        X = self.inputs[idx]
        Y = self.lut[X]

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        if Y.dim() == 0:  # Ensure Y is at least 1-dimensional
            Y = Y.unsqueeze(-1)

        return X, Y   
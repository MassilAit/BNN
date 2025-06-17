import torch
from torch.utils.data import Dataset
from itertools import product


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
    

class LogicDataSet(Dataset):
    """
    A dataset for Boolean functions defined by a truth table integer.

    Args:
        n (int): Number of input bits.
        tt_int (int): Integer encoding of the truth table output.
    """
    def __init__(self, n, tt_int):
        self.n = n
        self.inputs = list(product([0, 1], repeat=n))  # all possible input tuples
        self.outputs = [2 * b - 1 for b in self.tt_int_to_bits(tt_int, n)]
        self.n_data = 1 << n  # 2**n

    def __len__(self):
        return self.n_data
    
    def tt_int_to_bits(self, value, n):
        """Convert integer to truth table output bits."""
        length = 1 << n
        return [(value >> i) & 1 for i in range(length)]

    def __getitem__(self, idx):
        X = torch.tensor(self.inputs[idx], dtype=torch.float32)
        Y = torch.tensor([self.outputs[idx]], dtype=torch.float32)
        return X, Y
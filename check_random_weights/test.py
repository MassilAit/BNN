import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from model   import ContiniousdModel, BinarizedModel
from dataset import LogicDataSet
from train   import train_model

def make_loader(dataset, batch_size, seed):
    g = torch.Generator()
    g.manual_seed(seed)  # fixes the shuffle order across runs
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,     # keep 0 for simple reproducibility
        generator=g
    )

class BCEPlusMinusOne(nn.Module):
    """BCEWithLogitsLoss adapted for {-1, +1} targets."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets_pm1):
        targets_01 = (targets_pm1 + 1) / 2  # −1→0, +1→1
        return self.bce(logits, targets_01)
    




model = BinarizedModel(4, [5], bias=True, alpha = 0.8)

dataset = LogicDataSet(4, 27030)
    
loader = make_loader(dataset, 8, 1)

optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = BCEPlusMinusOne()



train_model(model, loader, optimizer,
                10_000, criterion,
                early_stop=True, delta=0.01, patience=100)
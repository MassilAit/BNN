from model import BinarizedModel, FullyBinarizedModel, ContiniousdModel
from dataset import LUTDataSet
from train import train_model, train_model_activations
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import os

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
    

XOR4 = {
    (-1, -1, -1, -1): -1,
    (-1, -1, -1,  1):  1,
    (-1, -1,  1, -1):  1,
    (-1, -1,  1,  1): -1,
    (-1,  1, -1, -1):  1,
    (-1,  1, -1,  1): -1,
    (-1,  1,  1, -1): -1,
    (-1,  1,  1,  1):  1,
    ( 1, -1, -1, -1):  1,
    ( 1, -1, -1,  1): -1,
    ( 1, -1,  1, -1): -1,
    ( 1, -1,  1,  1):  1,
    ( 1,  1, -1, -1): -1,
    ( 1,  1, -1,  1):  1,
    ( 1,  1,  1, -1):  1,
    ( 1,  1,  1,  1): -1
}




N_INPUTS = 4
BATCH_SIZE = 16
N_HIDDEN = [4,2]
bias = True
data= LUTDataSet(XOR4) 

is_bias = "bias" if bias else "no_bias"


train_loader= DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
N_EPOCHS = 10000


def test_output(data, model):

    model.eval()
    for X,Y_TRUE in data:
        with torch.no_grad():
            X=X.unsqueeze(0)
            logit = model(X).squeeze(0)

            Y_PRED=torch.sign(logit)
            
            if  Y_PRED.item() == 0:
                Y_PRED = torch.tensor(1., dtype=Y_TRUE.dtype)

            if Y_PRED.item() != Y_TRUE.item():
                return False
    
    return True

def run_one(i):
    model=BinarizedModel(N_INPUTS, N_HIDDEN, bias = bias)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = BCEPlusMinusOne()

    train_model(model,train_loader,optimizer,N_EPOCHS,criterion,True)

    if test_output(data, model):
        print("Loading model")
        save_path = f"Result/{N_INPUTS}_inputs/{N_HIDDEN}_neurons/{is_bias}/weights/{i}_weights.pth"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        return True
    
    print("Failed Learning")
    return False


total = 5
n = 0
    
while n < total:

    print(f"Testing model {n}")
    if run_one(n):
        n+=1

    
    


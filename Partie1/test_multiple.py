from model import BinarizedModel, FullyBinarizedModel, ContiniousdModel
from dataset import LUTDataSet
from train import train_model, train_model_activations
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import os


XOR3 = {
    (-1, -1, -1): -1,  
    (-1, -1,  1):  1,  
    (-1,  1, -1):  1,
    (-1,  1,  1): -1,  
    ( 1, -1, -1):  1,
    ( 1, -1,  1): -1,
    ( 1,  1, -1): -1,
    ( 1,  1,  1):  1   
}

XOR2 = {
    (-1, -1): -1,  
    (-1,  1):  1,  
    (1, -1):  1,
    (1,  1): -1,  
}

N_INPUTS = 3
BATCH_SIZE = 1
N_HIDDEN = 4
bias = False
data= LUTDataSet(XOR3) 

is_bias = "bias" if bias else "no_bias"


train_loader= DataLoader(data, batch_size=BATCH_SIZE)
N_EPOCHS = 1000


def test_output(data, model):

    for X,Y_TRUE in data:
        with torch.no_grad():
            Y_PRED = model(X)
            if not torch.equal(Y_PRED,Y_TRUE):
                return False
    
    return True

def run_one(i):
    model=BinarizedModel(N_INPUTS, [N_HIDDEN], bias = bias)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss() 

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

    
    


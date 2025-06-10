from model import BinarizedModel, FullyBinarizedModel
from dataset import LUTDataSet
from train import train_model, train_model_activations
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import os
import json


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

N_INPUTS = 2
BATCH_SIZE = 1
N_HIDDEN = 2
bias = False
data= LUTDataSet(XOR2) 

is_bias = "bias" if bias else "no_bias"


train_loader= DataLoader(data, batch_size=BATCH_SIZE)
N_EPOCHS = 1000


def output(data, model):
    r=""
    for X,_ in data:
        with torch.no_grad():
            Y_PRED = model(X)
            r+=str(int(Y_PRED))
            r+=","

    return r[:-1]

def load_to_json(list_of_dicts,path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    with open(path, 'w') as json_file:
        json.dump(list_of_dicts, json_file)
            

def run_one(dic):
    model=FullyBinarizedModel(N_INPUTS, [N_HIDDEN], bias = bias)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss() 

    params, grads, activation, loss_list = train_model_activations(model,train_loader,optimizer,N_EPOCHS,criterion,True, True)

    r=output(data,model)
    if r not in dic:
        dic[r]=0
    dic[r]+=1

    i=dic[r]

    save_path = f"Activations/FullyBinarized/{N_INPUTS}_inputs/{N_HIDDEN}_neurons/{is_bias}/{r}/weights/{i}_weights.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)

    load_to_json(params, f"Activations/FullyBinarized/{N_INPUTS}_inputs/{N_HIDDEN}_neurons/{is_bias}/{r}/params/{i}_params.json")
    load_to_json(grads, f"Activations/FullyBinarized/{N_INPUTS}_inputs/{N_HIDDEN}_neurons/{is_bias}/{r}/grads/{i}_grads.json")
    load_to_json(activation, f"Activations/FullyBinarized/{N_INPUTS}_inputs/{N_HIDDEN}_neurons/{is_bias}/{r}/activation/{i}_activation.json")

    return dic

total = 10
dic = {}

for i in range(total):

    print(f"Testing model {i}")
    dic=run_one(dic)

    
    


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
N_EPOCHS = 3


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
            

def run_one(model,name):
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss() 

    params, grads, activation, loss_list = train_model_activations(model,train_loader,optimizer,N_EPOCHS,criterion,True, True)

    load_to_json(params, f"Comparaison2/params/{name}_params.json")
    load_to_json(grads, f"Comparaison2/grads/{name}_grads.json")
    load_to_json(activation, f"Comparaison2/activation/{name}_activation.json")



r = "-1,-1,1,1"
i=1
save_path = f"Activations/Binarized/{N_INPUTS}_inputs/{N_HIDDEN}_neurons/{is_bias}/{r}/weights/{i}_weights.pth"

model_full=FullyBinarizedModel(N_INPUTS, [N_HIDDEN], bias = bias)
model_full.load_state_dict(torch.load(save_path))

model_simple=BinarizedModel(N_INPUTS, [N_HIDDEN], bias = bias)
model_simple.load_state_dict(torch.load(save_path))

run_one(model_full, "FullyBinarizedModel")
run_one(model_simple, "BinarizedModel")



    
    


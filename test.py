from model import BinarizedModel
from dataset import LUTDataSet
from train import train_model, train_model_activations
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import json 
import os


XOR= {(-1,-1):-1, (-1,1):1, (1,-1):1, (1,1):-1}
N_INPUTS = 2
BATCH_SIZE = 1
N_HIDDEN = 2
data=LUTDataSet(XOR)


train_loader= DataLoader(data, batch_size=BATCH_SIZE)
N_EPOCHS = 1000

def run_one(i):
    model=BinarizedModel(N_INPUTS, [N_HIDDEN])
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss() 

    params, grad, activations, loss = train_model_activations(model,train_loader,optimizer,N_EPOCHS,criterion,True,True)

    def load_to_json(list_of_dicts,path):
        folder = os.path.dirname(path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        with open(path, 'w') as json_file:
            json.dump(list_of_dicts, json_file)

    load_to_json(params, f"Result/{i}_params.json")
    load_to_json(grad, f"Result/{i}_grads.json")
    load_to_json(activations, f"Result/{i}_activations.json")

    result = []
    for X,Y_TRUE in data:
        with torch.no_grad():
            Y_PRED = model(X)
            result.append(Y_PRED==Y_TRUE)

    return result


#model=BinarizedModel(N_INPUTS, [N_HIDDEN])
#optimizer = optim.Adam(model.parameters(), lr=0.005)
#criterion = nn.MSELoss() 
#
#train_model(model,train_loader,optimizer,N_EPOCHS,criterion,True)

total = 10
bad = 0
    
for i in range(total):

    print(f"Testing model {i}")
    result = run_one(i)

    print(result)
    


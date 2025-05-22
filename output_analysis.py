import torch
from model import BinarizedModel
import pandas as pd

def find_unique_index(binary_str: str) -> str:
    if len(binary_str) < 2:
        return "x"

    # Count occurrences of '0' and '1'
    counts = {'0': 0, '1': 0}
    for c in binary_str:
        counts[c] += 1

    if counts['0'] == 1:
        return str(binary_str.index('0'))
    elif counts['1'] == 1:
        return str(binary_str.index('1'))
    else:
        return "x"



def tupleToTensor(a:tuple):
    return torch.tensor(a, dtype=torch.float32) 

inputs = [
    (-1, -1, -1),  
    (-1, -1,  1),  
    (-1,  1, -1),
    (-1,  1,  1),  
    ( 1, -1, -1),
    ( 1, -1,  1),
    ( 1,  1, -1),
    ( 1,  1,  1)   
]
XOR = {
    (-1, -1, -1): -1,  
    (-1, -1,  1):  1,  
    (-1,  1, -1):  1,
    (-1,  1,  1): -1,  
    ( 1, -1, -1):  1,
    ( 1, -1,  1): -1,
    ( 1,  1, -1): -1,
    ( 1,  1,  1):  1   
}

for i in range(len(inputs)):
    inputs[i]=tupleToTensor(inputs[i])


N = 20
N_INPUTS = 3
N_HIDDEN = 3
bias = False
is_bias = "bias" if bias else "no_bias"

model = BinarizedModel(N_INPUTS,[N_HIDDEN], bias = bias)

results = []

with torch.no_grad():
    for i in range(0, N):

        row = {"Model": i} 
        model.load_state_dict(torch.load(f"Result/{N_INPUTS}_inputs/{N_HIDDEN}_neurons/{is_bias}/weights/{i}_weights.pth"))

        #weights = model.state_dict()["model.2.weight"][0].tolist()
        #bias = float(model.state_dict()["model.2.bias"][0])

        outputs = [""]*N_HIDDEN
        for input in inputs : 
            model(input)
            activations = model.get_activations()["act_0"].tolist()
            for k in range(len(activations)):
                outputs[k]+=str(int(activations[k]>0))
        

        for idx, logic in enumerate(outputs):
            row[f"Neuron_{idx+1}"] = logic

        index=""
        for i, input in enumerate(XOR):
            index+="-"
            if XOR[input]==-1:
                index+="("
            for j ,logic in enumerate(outputs):
                index+=logic[i]

            if XOR[input]==-1:
                index+=")"


        #row[f"Bias"] = bias

        #for idx,weight in enumerate(weights) :
        #    row[f"Neuron_{idx+1}_weight"]=weight


        row["Summary"]=index[1:]
        results.append(row)



# Convert to DataFrame
df = pd.DataFrame(results)

# Save to CSV
df.to_csv(f"Result/{N_INPUTS}_inputs/{N_HIDDEN}_neurons/{is_bias}/summary.csv", index=False)

print(f"CSV saved")



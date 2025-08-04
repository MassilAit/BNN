import torch
from dataset import LogicDataSet
import torch.nn as nn
from model import BinarizedModel
from find_logic import minimise_one_ones
import re

def substitute_hidden(expr: str, hidden_exprs: list[str]) -> str:
    """
    Replace A, B, C, … in expr with their corresponding hidden expressions
    inside [ ] brackets, even if they appear concatenated like 'AB'.
    """
    out = ""
    for char in expr:
        if 'A' <= char <= 'Z':
            idx = ord(char) - ord('A')
            if idx < len(hidden_exprs):
                out += f"[{hidden_exprs[idx]}]"
            else:
                out += char
        else:
            out += char
    return out



def binary_list_to_int(bits):
    """
    bits: list[int] of 0/1
    returns: integer after reversing the list
    e.g., [1,0] → [0,1] → 1
    """
    bits=bits[::-1]
    value = 0
    for i, b in enumerate(bits):
        value |= (b << i)
    return value
# ───────────────────────────────────────── analyse whole network ──
def analyse_model(model: nn.Module, n_inputs: int, eval_ds: LogicDataSet, n_hidden):
  
    if len(n_hidden)==0 :
        minterms=[]
        with torch.no_grad():
            for i, (X, _) in enumerate(eval_ds):
                # predict output
                y_pred = torch.sign(model(X.unsqueeze(0))).item()
                if y_pred >=0 :
                    minterms.append(i)
        return minimise_one_ones(minterms, n_inputs)

            
    hidden_minterms = [[] for _ in range(n_hidden[0])]      
    output_true = set()
    output_false = set()  
    full_set = set(range(2**n_hidden[0]))                              

    with torch.no_grad():
        for i, (X, _) in enumerate(eval_ds):
            # predict output
            y_pred = torch.sign(model(X.unsqueeze(0))).item()
            #y_pred = 1 if y_pred > 0 else 0             # map -1/+1 → 0/1

            # hidden layer activations
            activ = model.get_activations()["act_0"][0].tolist()
            activ = [1 if a > 0 else 0 for a in activ]  # map -1/+1 → 0/1

            # record minterms for hidden neurons
            for h, h_val in enumerate(activ):
                if h_val == 1.0:
                    hidden_minterms[h].append(i)

            if y_pred ==1.0:
                output_true.add(binary_list_to_int(activ))
            else: 
                output_false.add(binary_list_to_int(activ))

    #print(f"Hidden minterms : {hidden_minterms}")

    dont_cares = full_set - (output_true | output_false)

    #print(f"Output true : {output_true}")
    #print(f"Output false : {output_false}")

    output_expression = minimise_one_ones(list(output_true), n_hidden[0], list(dont_cares))

    #print(f"Output expression : {output_expression}")

    l = []
    for minterms in hidden_minterms:
        #print(f"minterm expression : {minimise_one_ones(minterms, n_inputs) }")
        l.append(minimise_one_ones(minterms, n_inputs))

    return substitute_hidden(output_expression, l)


if __name__=="__main__":
    n_inputs=3
    tt_int=6
    n_hidden = [3]

    model= BinarizedModel(3,n_hidden)
    model.load_state_dict(torch.load(f"Result/{n_inputs}_inputs/{tt_int}_weights.pth"))
    data = LogicDataSet(n_inputs, tt_int)

    print(analyse_model(model,n_inputs, data, n_hidden))



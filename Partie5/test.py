import torch
from dataset import LogicDataSet
import torch.nn as nn
from model import BinarizedModel
from find_logic import minimise_one_ones
import re

import re

def substitute_hidden(expr: str, hidden_exprs: list[str]):
    """
    Replace A, B, C, … in expr with their corresponding hidden expressions
    inside [ ] brackets.

    hidden_exprs: list of strings (indexed: 0 → A, 1 → B, …)

    Example:
        expr = "(A + B')(C')"
        hidden_exprs = ["X", "Y", "Z"]
        → "([X] + [Y]')([Z]')"
    """
    def repl(match):
        var = match.group(0)
        idx = ord(var) - ord('A')
        if idx < len(hidden_exprs):
            return f"[{hidden_exprs[idx]}]"
        else:
            return var  # leave it unchanged if no mapping
    
    # Substitute only standalone variables A-Z
    substituted = re.sub(r"\b[A-Z]\b", repl, expr)
    return substituted


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
    """
    Returns a dict:
        {
          'hidden': [ expr_dict_neuron0, expr_dict_neuron1, ... ],
          'output': expr_dict_output
        }
    Each expr_dict_* is what `minimise_from_bits` returns.
    """

    hidden_minterms = [[] for _ in range(n_hidden)]      
    output_true = set()
    output_false = set()  
    full_set = set(range(2**n_inputs))                              

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


    dont_cares = full_set - (output_true | output_false)

    output_expression = minimise_one_ones(list(output_true), n_inputs, list(dont_cares))
    print(output_expression)

    l = []
    for minterms in hidden_minterms:
        print(minimise_one_ones(minterms, n_inputs))
        l.append(minimise_one_ones(minterms, n_inputs))

    return substitute_hidden(output_expression, l)


model=BinarizedModel(3,[3])
model.load_state_dict(torch.load(f"Result/{3}_inputs/{6}_weights.pth"))

eval_ds=LogicDataSet(3,6)
print(analyse_model(model,3,eval_ds, 3))


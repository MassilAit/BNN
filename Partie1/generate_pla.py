import torch
from model import BinarizedModel

N = 0
N_INPUTS = 3
N_HIDDEN = 3
bias = False
is_bias = "bias" if bias else "no_bias"

def input_to_bin(t):
    """Convert a tuple of -1/+1 to a binary string of 0/1."""
    return ''.join(['1' if v == 1 else '0' for v in t])

def generate_pla_with_defaults(all_inputs: list, mapping: dict, num_inputs: int) -> str:
    """
    Generate a PLA string from a partial mapping, filling missing entries with don't cares.
    
    Args:
        all_inputs: list of all input tuples to consider (e.g., from itertools.product)
        mapping: dictionary mapping some of those tuples to 0, 1, or '-'
        num_inputs: number of input bits
    
    Returns:
        PLA file content as string
    """
    lines = [f".i {num_inputs}", ".o 1", f".p {len(all_inputs)}"]
    
    for input_tuple in all_inputs:
        bin_input = input_to_bin(input_tuple)
        output = mapping.get(input_tuple, "-")
        lines.append(f"{bin_input} {str(output)}")
    
    lines.append(".e")
    return "\n".join(lines)

def tupleToTensor(a:tuple):
    return torch.tensor(a, dtype=torch.float32) 

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




model = BinarizedModel(N_INPUTS,[N_HIDDEN], bias = bias)
model.load_state_dict(torch.load(f"Result/{N_INPUTS}_inputs/{N_HIDDEN}_neurons/{is_bias}/weights/{N}_weights.pth"))
hidden=[{} for _ in range(N_HIDDEN)]
output_tt={}
for input, output in XOR.items() : 
            model(tupleToTensor(input))
            activations = model.get_activations()["act_0"].int().tolist()
            output_tt[tuple(activations)] = int(output>0)
            for k in range(len(activations)):
                hidden[k][input]=int(activations[k]>0)


print("Hidden layer : ")
for x in hidden:
    for input, output in x.items():
       print(f"{input} : {output}")


print("=================================")
print("Output Layer : ")
for input, output in output_tt.items():
       print(f"{input} : {output}")


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

# 2. Write hidden layer .pla files
for i in range(N_HIDDEN):
    content = generate_pla_with_defaults(inputs, hidden[i], N_INPUTS)
    with open(f"hidden{i+1}.pla", "w") as f:
        f.write(content)
    print(f"✅ saved hidden{i+1}.pla")


content = generate_pla_with_defaults(inputs, output_tt, N_HIDDEN)

with open("output.pla", "w") as f:
    f.write(content)
print("✅ saved output.pla")






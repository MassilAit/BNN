from check_activation import *
from model import BinarizedModel, BinaryLinear
from dataset import LUTDataSet
import copy


def disable_hidden_neuron(model, layer_idx, neuron_idx):
    """
    Disable neuron `neuron_idx` in hidden layer `layer_idx` of a BinarizedModel.
    Assumes BinaryLinear layers and BinaryActivation alternate in model.model.
    """
    # Index of BinaryLinear layers in Sequential: 0, 2, 4, ...
    linear_idx = layer_idx * 2
    next_linear_idx = linear_idx + 2

    curr_layer = model.model[linear_idx]
    next_layer = model.model[next_linear_idx]

    # Sanity check
    assert isinstance(curr_layer, BinaryLinear), "Expected BinaryLinear at current layer"
    assert isinstance(next_layer, BinaryLinear), "Expected BinaryLinear at next layer"

    # Zero out the neuron's incoming and outgoing weights
    with torch.no_grad():
        next_layer.weight[:, neuron_idx] = 0       # outgoing weights
        curr_layer.weight[neuron_idx, :] = 0       # incoming weights
        curr_layer.bias[neuron_idx] = 0            # bias


def test_output(data, model):

    for X,Y_TRUE in data:
        with torch.no_grad():
            Y_PRED = model(X)
            if not torch.equal(Y_PRED,Y_TRUE):
                return False
    
    return True

def print_output(data, model:BinarizedModel):
    
    result=[]
    activation=[]
    for X, _ in data:
        with torch.no_grad():
            result.append(int(model(X)))
            activation.append(model.get_activations())
            
    
    print(result)
    return activation



N = 0
N_INPUTS = 3
N_HIDDEN = 3
bias = False
is_bias = "bias" if bias else "no_bias"

XOR= {(-1,-1):-1, (-1,1):1, (1,-1):1, (1,1):-1}
data=LUTDataSet(XOR)

model = BinarizedModel(N_INPUTS,[N_HIDDEN],bias=bias)
model.load_state_dict(torch.load(f"Result/{N_INPUTS}_inputs/{N_HIDDEN}_neurons/{is_bias}/weights/{N}_weights.pth"))



print(model.state_dict())
#activation_before = copy.deepcopy(print_output(data, model))



#disable_hidden_neuron(model, 0, 0)
#print(model.state_dict())
#activation_after = print_output(data, model)
#
#for i in range(len(activation_after)):
#    print(activation_before[i])
#    print(activation_after[i])
#    print()










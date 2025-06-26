import torch
import torch.nn as nn
import torch.nn.functional as F


class BinarySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        out = x.sign()
        out[out == 0] = 1   # treat 0 as +1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # STE: pass gradient where |x| ≤ 1, block elsewhere
        grad_input = grad_output.clone()
        alpha=0.8
        grad_input[x.abs() > alpha] = 0    # => this is problematic 
        return grad_input

binary_sign = BinarySign.apply  # convenience handle

# ────────────────────────────────────────────────
# 2.  Binary linear layer (weights binarised each forward)
# ────────────────────────────────────────────────
class BinaryLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features= in_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        w_bin = binary_sign(self.weight)          # binarise weights
        z = F.linear(x, w_bin, self.bias)
        return z

# ────────────────────────────────────────────────
# 3.  Binary activation wrapper (sign with STE)
# ────────────────────────────────────────────────
class BinaryActivation(nn.Module):
    def forward(self, x):
        return binary_sign(x)

# ────────────────────────────────────────────────
# 4.  Fully-binarised network
# ────────────────────────────────────────────────
class FullyBinarizedModel(nn.Module):
    """
    Binarized network that records pre-activations and activations during forward pass.
    """
    def __init__(self, n_inputs, hidden_layers, bias = True):
        super().__init__()

        self.model = nn.Sequential()
        self.activations = {}
        self.hooks = []

        current_in = n_inputs
        layer_idx = 0

        for h in hidden_layers:
            linear = BinaryLinear(current_in, h, bias = bias)
            act = BinaryActivation()

            self.model.add_module(f"linear_{layer_idx}", linear)
            self.model.add_module(f"act_{layer_idx}", act)

            current_in = h
            layer_idx += 1

        # Output layer
        self.model.add_module(f"linear_{layer_idx}", BinaryLinear(current_in, 1, bias=bias))
        self.model.add_module(f"act_{layer_idx}", BinaryActivation())

        self.register_hooks()

    def register_hooks(self):
        """Register hooks to store pre-activations and activations"""
        def hook_fn(module, input, output, name):
            self.activations[name] = output.detach()

        for name, layer in self.model.named_children():
            if isinstance(layer, BinaryLinear) or isinstance(layer, BinaryActivation):
                hook = layer.register_forward_hook(lambda m, i, o, name=name: hook_fn(m, i, o, name))
                self.hooks.append(hook)

    def forward(self, x):
        return self.model(x)

    def get_activations(self):
        return self.activations
    

class BinarizedModel(nn.Module):
    """
    Binarized network that records pre-activations and activations during forward pass.
    """
    def __init__(self, n_inputs, hidden_layers, bias = True):
        super().__init__()

        self.model = nn.Sequential()
        self.activations = {}
        self.hooks = []

        current_in = n_inputs
        layer_idx = 0

        for h in hidden_layers:
            linear = nn.Linear(current_in, h, bias = bias)
            #bn = nn.BatchNorm1d(h,)
            act = BinaryActivation()

            self.model.add_module(f"linear_{layer_idx}", linear)
            #self.model.add_module(f"batchNorm_{layer_idx}", bn)
            self.model.add_module(f"act_{layer_idx}", act)

            current_in = h
            layer_idx += 1

        # Output layer
        self.model.add_module(f"linear_{layer_idx}", nn.Linear(current_in, 1, bias=bias))
        #self.model.add_module(f"act_{layer_idx}", BinaryActivation())

        self.register_hooks()

    def register_hooks(self):
        """Register hooks to store pre-activations and activations"""
        def hook_fn(module, input, output, name):
            self.activations[name] = output.detach()

        for name, layer in self.model.named_children():
            if isinstance(layer, nn.Linear) or isinstance(layer, BinaryActivation):
                hook = layer.register_forward_hook(lambda m, i, o, name=name: hook_fn(m, i, o, name))
                self.hooks.append(hook)

    def forward(self, x):
        return self.model(x)

    def get_activations(self):
        return self.activations
    

class ContiniousdModel(nn.Module):
    """
    Binarized network that records pre-activations and activations during forward pass.
    """
    def __init__(self, n_inputs, hidden_layers, bias = True):
        super().__init__()

        self.model = nn.Sequential()
        self.activations = {}
        self.hooks = []

        current_in = n_inputs
        layer_idx = 0

        for h in hidden_layers:
            linear = nn.Linear(current_in, h, bias = bias)
            act = nn.Tanh()

            self.model.add_module(f"linear_{layer_idx}", linear)
            self.model.add_module(f"act_{layer_idx}", act)

            current_in = h
            layer_idx += 1

        # Output layer
        self.model.add_module(f"linear_{layer_idx}", nn.Linear(current_in, 1, bias=bias))
        #self.model.add_module(f"act_{layer_idx}", nn.Tanh())

        self.register_hooks()

    def register_hooks(self):
        """Register hooks to store pre-activations and activations"""
        def hook_fn(module, input, output, name):
            self.activations[name] = output.detach()

        for name, layer in self.model.named_children():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.Tanh):
                hook = layer.register_forward_hook(lambda m, i, o, name=name: hook_fn(m, i, o, name))
                self.hooks.append(hook)

    def forward(self, x):
        return self.model(x)

    def get_activations(self):
        return self.activations



import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn

# ────────────────────────────────────────────────
# 1) STE with configurable alpha
# ────────────────────────────────────────────────
class BinarySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha: float):
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        out = x.sign()
        out[out == 0] = 1   # treat 0 as +1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output.clone()
        grad_input[x.abs() > alpha] = 0
        # return grad for each Tensor input to forward; alpha is non-tensor ⇒ None
        return grad_input, None

binary_sign = BinarySign.apply


# ────────────────────────────────────────────────
# 2) Binary activation module that *stores* alpha
#    (as a buffer so it saves/loads & moves with the model)
# ────────────────────────────────────────────────
class BinaryActivation(nn.Module):
    def __init__(self, alpha: float = 0.8):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, x):
        return binary_sign(x, float(self.alpha))


# ────────────────────────────────────────────────
# 3) Binarized model that passes alpha to each activation
# ────────────────────────────────────────────────
class BinarizedModel(nn.Module):
    def __init__(self, n_inputs, hidden_layers, bias=True, alpha: float = 0.8):
        super().__init__()
        self.model = nn.Sequential()
        self.activations = {}
        self.hooks = []

        current_in = n_inputs
        layer_idx = 0
        for h in hidden_layers:
            self.model.add_module(f"linear_{layer_idx}", nn.Linear(current_in, h, bias=bias))
            self.model.add_module(f"act_{layer_idx}", BinaryActivation(alpha=alpha))
            current_in = h
            layer_idx += 1

        self.model.add_module(f"linear_{layer_idx}", nn.Linear(current_in, 1, bias=bias))
        self.register_hooks()

    def register_hooks(self):
        def hook_fn(module, input, output, name):
            self.activations[name] = output.detach()
        for name, layer in self.model.named_children():
            if isinstance(layer, (nn.Linear, BinaryActivation)):
                hook = layer.register_forward_hook(lambda m, i, o, name=name: hook_fn(m, i, o, name))
                self.hooks.append(hook)

    def forward(self, x):
        return self.model(x)

    def get_activations(self):
        return self.activations

    # Optional: change alpha at runtime for all BinaryActivation layers
    def set_alpha(self, alpha: float):
        for m in self.model.modules():
            if isinstance(m, BinaryActivation):
                m.alpha.fill_(float(alpha))

    

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



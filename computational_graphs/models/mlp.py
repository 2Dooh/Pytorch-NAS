from torch.nn import Module, Linear, ModuleList, ModuleDict, Sequential
import torch.nn as nn

class MultiLayerPerceptron(Module):
    def __init__(self, 
                 layers_dim, 
                 activations, 
                 **kwargs):
        super(MultiLayerPerceptron, self).__init__()
        self.linears = ModuleList(
            [Linear(in_features, out_features) \
                for in_features, out_features in zip(layers_dim, layers_dim[1:])]
        )

        self.activations = ModuleDict([
            [key, func] for key, func in zip(
                activations, 
                map(lambda f: getattr(nn, f, None)(), activations)
            )
        ])
        self.list_activations = activations

    def forward(self, x):
        for linear, act in zip(self.linears, self.list_activations):
            x = linear(x)
            f = self.activations[act]
            x = f(x) if f else x

        return x
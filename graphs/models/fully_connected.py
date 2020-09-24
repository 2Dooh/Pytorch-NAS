from torch.nn import Module, Linear
import torch

class FullyConnected(Module):
    def __init__(self, layers_dim, activations, **kwargs):
        super(FullyConnected, self).__init__()
        self.linears = []
        self.activations = []
        for i in range(len(layers_dim)-1):
            self.__setattr__('linear' + str(i+1), Linear(layers_dim[i], 
                                                         layers_dim[i+1]))
            self.activations += [getattr(torch, activations[i], None)]

    def forward(self, x):
        for i, activation in enumerate(self.activations):
            linear = self.__getattr__('linear' + str(i+1))
            x = linear(x) if activation is None else activation(linear(x))
        return x

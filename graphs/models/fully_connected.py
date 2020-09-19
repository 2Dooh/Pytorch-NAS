from torch.nn import Module, Linear
import torch

class FullyConnected(Module):
    def __init__(self, config):
        super(FullyConnected, self).__init__()
        self.linears = []
        self.activations = []
        for i in range(len(config.layers_dim)-1):
            self.__setattr__('linear' + str(i+1), Linear(config.layers_dim[i], 
                                                         config.layers_dim[i+1]))
            self.activations += [getattr(torch, config.activations[i], None)]

    def forward(self, x):
        for i, activation in enumerate(self.activations):
            linear = self.__getattr__('linear' + str(i+1))
            x = linear(x) if activation is None else activation(linear(x))
        return x

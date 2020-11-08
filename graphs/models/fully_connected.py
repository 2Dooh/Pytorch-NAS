from torch.nn import Module, Linear, ModuleList, ModuleDict, Sequential
import torch
import torch.nn.functional as F
class FullyConnected(Module):
    def __init__(self, 
                 layers_dim, 
                 activations, 
                 **kwargs):
        super(FullyConnected, self).__init__()
        self.__name__ = 'FullyConnected'
        self.linears = ModuleList()
        for input_size, output_size in zip(layers_dim, layers_dim[1:]):
            self.linears += [Linear(input_size, output_size)]
        self.activations = activations

    def forward(self, x):
        for activation, linear in zip(self.activations, self.linears):
            activation = getattr(F, activation, None)
            x = activation(linear(x)) if activation else linear(x)
        return x
    
    # Constructor
    # def __init__(self, layers_dim, **kwargs):
    #     super(FullyConnected, self).__init__()
    #     self.hidden = ModuleList()
    #     for input_size, output_size in zip(layers_dim, layers_dim[1:]):
    #         self.hidden.append(Linear(input_size, output_size))
    
    # # Prediction
    # def forward(self, activation):
    #     L = len(self.hidden)
    #     for (l, linear_transform) in zip(range(L), self.hidden):
    #         if l < L - 1:
    #             activation = F.relu(linear_transform(activation))    
    #         else:
    #             activation = linear_transform(activation)
    #     return activation

from torch import optim
from torch.nn import Module, Linear
import torch
import torch.nn.functional as F

class LogisticRegression(Module):
    def __init__(self, 
                 input_size, 
                 output_size, 
                 activation, 
                 **kwargs):
        super(LogisticRegression, self).__init__()
        self.__name__ = 'LogisticRegression'
        self.linear = Linear(input_size, output_size)
        self.activation = activation

    def forward(self, x):
        activation = getattr(F, self.activation, None)
        x = activation(self.linear(x)) if activation else self.linear(x)
        return x
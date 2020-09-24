from torch import optim
from torch.nn import Module, Linear
import torch

class LogisticRegression(Module):
    def __init__(self, input_size, activation, **kwargs):
        super(LogisticRegression, self).__init__()
        self.linear = Linear(input_size, 1)
        self.activation = getattr(torch, activation, None)

    def forward(self, x):
        y_pred = self.activation(self.linear(x))
        return y_pred
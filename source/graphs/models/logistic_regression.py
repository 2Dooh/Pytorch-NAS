from torch import optim
from torch.nn import Module, Linear
import torch

class LogisticRegression(Module):
    def __init__(self, config):
        super(LogisticRegression, self).__init__()
        self.linear = Linear(config.input_size, 1)
        self.activation = getattr(torch, config.activation, None)

    def forward(self, x):
        y_pred = self.activation(self.linear(x))
        return y_pred
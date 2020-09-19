from torch import optim
from torch.nn import Module, Linear

class LinearRegression(Module):
    def __init__(self, config):
        super(LinearRegression, self).__init__()
        self.linear = Linear(config.input_size, config.output_size)

    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat

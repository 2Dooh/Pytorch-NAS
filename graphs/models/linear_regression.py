from torch import optim
from torch.nn import Module, Linear

class LinearRegression(Module):
    def __init__(self, input_size, output_size, **kwargs):
        super(LinearRegression, self).__init__()
        self.linear = Linear(input_size, output_size)

    def forward(self, x):
        y_hat = self.linear(x)
        return y_hat

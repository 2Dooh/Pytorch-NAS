import computational_graphs.estimators.problems.single_obj.so_problem as base

import numpy as np
from numpy import pi

import torch

class Rastrigin(base.SingleObjectiveProblem):
    def __init__(self, n_params=2, **kwargs):
        super().__init__(n_params=n_params,
                         constraints=0,
                         n_obj=1,
                         type=torch.float,
                         multi_dims=True,
                         **kwargs)
        xl = torch.ones(self.n_params, device=self.device) * -5.12
        xu = torch.ones(self.n_params, device=self.device) * -5.12
        self.domain = (xl, xu)
        # self._pareto_set = np.zeros((1, n_params), dtype=self.param_type)
        # self._pareto_front = 0
        self.opt = torch.min
        self.argopt = torch.argmin
        self.A = 10

    ## Overide Methods ##
    @property
    def X_opt(self):
        return torch.zeros(1, self.n_samples).type(self.type)
    @property
    def Y_opt(self):
        return 0

    def _f(self, X):
        f = self.A*X.size(1) + (X**2 - self.A*torch.cos(2*pi*X)).sum(dim=1)
        return f

    @staticmethod
    def _compare(Y1, Y2):
        return Y1 < Y2

class ModifiedRastrigin(base.SingleObjectiveProblem):
    def __init__(self, n_params=2, **kwargs):
        super().__init__(n_params,
                         n_constraints=0,
                         type=np.double,
                         multi_dims=True)
        xl = torch.ones(self.n_params,) * 0
        xu = torch.ones(self.n_params,) * 1
        self.domain = (xl, xu)
        # self._pareto_set = np.zeros((1, n_params), dtype=self.param_type)
        # self._pareto_front = 0
        self.opt = torch.min
        self.argopt = torch.argmin
        self.A = 10
        self.k = torch.Tensor(
            [1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 1, 4]
        )
        if self.n_params == 2:
            self.k = torch.Tensor([3, 4])
        else:
            self.k = torch.ones(self.n_params, 1)

    ## Overide Methods ##
    def _f(self, X):
        self.k = self.k.to(X.device)
        try:
            f = -(10 + 9*torch.cos(2*pi*X*self.k)).sum()
        except:
            f = -(10 + 9*torch.cos(2*pi*X*self.k[:, None])).sum(axis=0)
        return f

    def _compare(y1, y2):
        return y1 <= y2
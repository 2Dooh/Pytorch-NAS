
import numpy as np

import computational_graphs.operators.base as base


def repair_out_of_bounds_manually(X, xl, xu):
    if xl is not None:
        xl = np.ones_like(X) * xl
        # xl = torch.repeat(xl[None, :], X.size(0), dim=0)
        X[X < xl] = xl[X < xl]

    if xu is not None:
        xu = np.ones_like(X) * xu
        # xu = torch.repeat(xu[None, :], X.size(0), dim=0)
        X[X > xu] = xu[X > xu]
        
    return X


def repair_out_of_bounds(problem, X):
    return repair_out_of_bounds_manually(X, *problem.domain)


class OutOfBoundsRepair(base.OperatorBase):
    def __call__(self, problem, pop, **kwargs):
        X = pop
        repaired_X = repair_out_of_bounds(problem ,X)
        return repaired_X

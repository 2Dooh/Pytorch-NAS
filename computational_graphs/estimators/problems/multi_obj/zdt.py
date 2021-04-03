from numpy.lib.function_base import vectorize
import computational_graphs.estimators.problems.multi_obj.mo_problem as base

from numpy import pi
import numpy as np

class ZDT4(base.MultiObjectiveProblem):
    def __init__(self, n_params=10, **kwargs):
        super().__init__(n_params=n_params,
                         n_obj=2,
                         constraints=0,
                         type=np.float,
                         vectorized=True,
                         **kwargs)
        xl = np.ones(self.n_params) * -5
        xu = np.ones(self.n_params) * 5
        xl[0], xu[0] = 0, 1
        self.domain = (xl, xu)
    
    def _f(self, X):
        f1 = self.__f1(X)
        f2 = self.__f2(f1, X)
        return np.concatenate([f1[:, None], f2[:, None]], axis=1)
    
    @staticmethod
    def __f1(X):
        return X[:, 0]

    def __f2(self, f1, X):
        g = self.__g(X)
        return g * self.__h(f1, g)

    @staticmethod
    def __g(X):
        g = (1 + (10*(X.shape[1]-1)) + \
            (X[:, 1:]**2 - 10*np.cos(4*pi*X[:, 1:])).sum(axis=1))
        return g

    @staticmethod
    def __h(f1, g):
        return 1 - np.power(f1/g, 0.5)

    @staticmethod
    def _compare(y1, y2):
        not_dominated = y1 <= y2
        dominate = y1 < y2
        return not_dominated.all() and True in dominate

    @staticmethod
    def _compare_old(y1, y2):
        return (y1[0] <= y2[0] and y1[1] <= y2[1]) and \
               (y1[0] < y2[0] or y1[1] < y2[1])

    @staticmethod
    def _compare_vectorized(Y1, Y2):
        not_dominated = Y1 <= Y2
        dominate = Y1 < Y2

        pareto_dominants = np.logical_and(
            not_dominated.all(axis=1),
            dominate.any(axis=1)
        ) 
        return pareto_dominants

    def _calc_pareto_front(self, n_points):
        X1 = np.linspace(0, 1, n_points)
        X2 = 1 - np.sqrt(X1)
        return (X1, X2)
import numpy as np

import computational_graphs.operators.base as base

from abc import abstractmethod

class CrossoverBase(base.OperatorBase):
    def __init__(self, problem, n_parents, n_offs, prob=0.9, **kwargs):
        super().__init__(problem, **kwargs)
        self.prob = prob
        self.n_parents = n_parents
        self.n_offs = n_offs

    def __call__(self, pop, parents, **kwargs):
        return super().__call__(**kwargs)

    @abstractmethod
    def _call(self, **kwargs):
        raise NotImplementedError
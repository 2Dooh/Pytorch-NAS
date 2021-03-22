import computational_graphs.operators.base as base

import numpy as np

class BitFlipMutator(base.OperatorBase):
    def __init__(self, prob=None, **kwargs):
        super().__init__()
        self.prob = prob

    def __call__(self, pop, **kwargs):
        self.prob = 1/self.problem.n_params if self.prob is None else self.prob
        R = np.random.random(pop.shape)
        offs = pop.copy()
        mutation_points = pop[R < self.prob].astype(np.bool)
        offs[R < self.prob] = (mutation_points == False).astype(self.problem.type)
        return offs

    
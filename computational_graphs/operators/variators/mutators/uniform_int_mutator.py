import numpy as np

import computational_graphs.operators.base as base

from utils.evo_alg.denormalization import denormalize

class UniformIntMutator(base.OperatorBase):
    def __init__(self, prob=None, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob

    def __call__(self, pop, **kwarg):
        (xl, xu) = self.problem.domain
        xl, xu = xl.min(), xu.max()
        self.prob = 1/self.problem.n_params if self.prob is None else self.prob
        R = np.random.rand(*pop.shape)
        offs = pop.copy()
        n_mut = len(R[R < self.prob])
        mutation_vector = np.random.rand(n_mut)
        mutation_vector = denormalize(mutation_vector, xl, xu).astype(self.problem.type)

        offs[R < self.prob] = mutation_vector
        return offs

        
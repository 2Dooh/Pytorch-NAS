import computational_graphs.operators.base as base
from computational_graphs.operators.repairers.out_of_bounds_repair import repair_out_of_bounds_manually 

import numpy as np

class BitFlipMutator(base.OperatorBase):
    def __init__(self, prob=None, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob

    def __call__(self, pop, **kwargs):
        self.prob = 1/self.problem.n_params if self.prob is None else self.prob
        R = np.random.random(pop.shape)
        offs = pop.copy()
        mutation_points = pop[R < self.prob].astype(np.bool)
        offs[R < self.prob] = (mutation_points == False).astype(self.problem.type)
        offs = repair_out_of_bounds_manually(offs, *self.problem.domain)
        return offs

    
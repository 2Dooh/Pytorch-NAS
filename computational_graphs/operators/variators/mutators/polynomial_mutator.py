import computational_graphs.operators.base as base

import numpy as np


class PolynomialMutator(base.OperatorBase):
    def __init__(self, eta=20, prob=None, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta
        self.prob = prob
        if self.prob is None:
            self.prob = 1 / self.problem.n_params

    def __call__(self, pop, **kwargs):
        (XL, XU) = self.problem.domain
        pop = pop.copy()
        
        indices = list(range(pop.shape[0]))
        for idx in indices:
            R = np.random.rand(pop.shape[1])
            mutation_points = R <= self.prob
            x = pop[idx][mutation_points]
            xl, xu = XL[mutation_points], XU[mutation_points]

            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = np.random.rand(x.shape[0])
            mut_pow = 1 / (self.eta + 1)
            delta_q = self.__calc_delta_q(rand, mut_pow, delta_1, delta_2)

            x = x + delta_q * (xu - xl)
            x = np.minimum(np.maximum(x, xl), xu).astype(self.problem.type)
            
            pop[idx][mutation_points] = x
        return pop

    def __calc_delta_q(self, rand, mut_pow, delta_1, delta_2):
        delta_q = np.empty_like(rand)

        mask = rand < 0.5
        xy = 1 - delta_1[mask]
        val = 2 * rand[mask] + (1 - 2 * rand[mask]) * xy ** (self.eta + 1)
        delta_q[mask] = val ** mut_pow - 1

        mask_not = rand >= 0.5
        xy = 1 - delta_2[mask_not]
        val = 2 * (1 - rand[mask_not]) + 2 * (rand[mask_not] - 0.5) * xy ** (self.eta + 1)
        delta_q[mask_not] = 1 - val ** mut_pow

        return delta_q
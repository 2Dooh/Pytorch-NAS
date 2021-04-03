import computational_graphs.operators.base as base


import numpy as np

import numpy.random as random
#import random

class SimulatedBinaryCrossover(base.OperatorBase):
    def __init__(self, eta=15, prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.eta = eta
        self.prob = prob

    def __call__(self, pop, **kwargs):
        # (n_inds, n_params) = pop.pop.shape
        (XL, XU) = self.problem.domain
        
        indices = np.arange(pop.shape[0])

        offs = []
        random.shuffle(indices)

        for i in range(0, pop.shape[0], 2):
            idx1, idx2 = indices[i], indices[i+1]
            offs1, offs2 = pop[idx1].copy(), pop[idx2].copy()

            R = np.random.rand(pop.shape[1])
            diffs = np.abs(offs1 - offs2)
            crossover_points = np.logical_and(R <= self.prob, diffs > 1e-14)
            xl, xu = XL[crossover_points], XU[crossover_points]
            x1 = np.minimum(offs1[crossover_points], offs2[crossover_points])
            x2 = np.maximum(offs1[crossover_points], offs2[crossover_points])
            rand = np.random.rand(*x2.shape)

            beta = 1 + (2 * (x1-xl)/x2-xl)
            beta_q = self.__calc_beta_q(rand, beta)
            c1 = 0.5 * (x1+x2 - beta_q*(x2-x1))

            beta = 1 + (2 * (xu-x2)/x2-x1)
            beta_q = self.__calc_beta_q(rand, beta)
            c2 = 0.5 * (x1+x2 + beta_q*(x2-x1))

            c1 = np.minimum(np.maximum(c1, xl), xu)
            c2 = np.minimum(np.maximum(c2, xl), xu)

            r = np.random.rand(*rand.shape)
            c1[r <= 0.5], c2[r <= 0.5] = c2[r <= 0.5], c1[r <= 0.5]
            offs1[crossover_points] = c1
            offs2[crossover_points] = c2

            offs += [offs1[None, :], offs2[None, :]]  
        
        return np.array(offs).reshape(pop.shape)
        # return np.reshape(offs, pop.pop.shape)

    def __calc_beta_q(self, rand, beta):
        alpha = 2 - beta**-(self.eta+1)
        beta_q = np.empty_like(beta)
        mask, mask_not = rand <= 1/alpha, rand > 1/alpha
        
        beta_q[mask] = (rand[mask] * alpha[mask]) ** (1 / (self.eta + 1))
        beta_q[mask_not] = (1 / (2 - rand[mask_not] * alpha[mask_not])) ** (1 / (self.eta + 1))
        
        return beta_q

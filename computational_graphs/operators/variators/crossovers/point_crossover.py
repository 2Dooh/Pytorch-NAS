  
import numpy as np

import computational_graphs.operators.base as base


class PointCrossover(base.OperatorBase):
    def __init__(self, **kwargs):
        super().__init__()

    def __call__(self, pop, **kwargs):
        (n_inds, n_params) = pop.shape
        indices = np.arange(n_inds)

        offs = []
        np.random.shuffle(indices)

        for i in range(0, n_inds, 2):
            idx1, idx2 = indices[i], indices[i+1]
            offs1, offs2 = pop[idx1].copy(), pop[idx2].copy()

            point = np.random.randint(low=0, high=n_params-1)
            offs1[:point], offs2[:point] = offs2[:point], offs1[:point].copy()

            offs.append(offs1)
            offs.append(offs2)
        
        return np.reshape(offs, pop.shape)
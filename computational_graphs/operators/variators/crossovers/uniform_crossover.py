import numpy as np
import computational_graphs.operators.base as base


class UniformCrossover(base.OperatorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_parents = self.n_offs = 2

    def __call__(self, pop, **kwargs):
        random_mask = np.random.rand(*pop.shape)
        _pop = pop.copy()
        _pop[0][random_mask] = _pop[1][random_mask]
        
        # indices = np.arange(n_inds)

        # offs = []
        # np.random.shuffle(indices)

        # for i in range(0, n_inds, 2):
        #     idx1, idx2 = indices[i], indices[i+1]
        #     offs1, offs2 = pop[idx1].copy(), pop[idx2].copy()

        #     points = np.random.uniform(low=0, high=1, size=(n_params,))
        #     offs1[points < self.prob], offs2[points < self.prob] = offs2[points < self.prob], offs1[points < self.prob]

        #     offs.append(offs1)
        #     offs.append(offs2)
        
        # return np.reshape(offs, pop.shape)
import computational_graphs.operators.base as base

import numpy as np

class MB1X(base.OperatorBase):
    def __init__(self, prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob

    def __call__(self, pop, **kwargs):
        if hasattr(self.problem, 'model'):
            model = self.problem.model
        else:
            self.logger.error('Model not found!') 
        (n_inds, n_params) = pop.shape
        indices = np.arange(n_inds)
        n_groups = len(model)


        offs = []
        np.random.shuffle(indices)

        for i in range(0, n_inds, 2):
            idx1, idx2 = indices[i], indices[i+1]
            offs1, offs2 = pop[idx1].copy(), pop[idx2].copy()

            point = np.random.randint(low=0, high=n_groups-1)
            for idx, group in enumerate(model):
                if idx < point:
                    offs1[group], offs2[group] = offs2[group].copy(), offs1[group]

            offs.append(offs1)
            offs.append(offs2)
        
        return np.reshape(offs, pop.shape)
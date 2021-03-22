import computational_graphs.operators.base as base

import numpy as np

class MBUniformCrossover(base.OperatorBase):
    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def __call__(self, ga):
        if hasattr(ga, 'model'):
            model = ga.model
        elif hasattr(ga.problem, 'problem_model'):
            model = ga.problem.problem_model
        else:
            raise Exception('Model not found!') 
        (n_inds, n_params) = ga.pop.shape
        indices = np.arange(n_inds)
        n_groups = len(model)

        offs = []
        np.random.shuffle(indices)

        for i in range(0, n_inds, 2):
            idx1, idx2 = indices[i], indices[i+1]
            offs1, offs2 = ga.pop[idx1].copy(), ga.pop[idx2].copy()

            points = np.random.uniform(low=0, high=1, size=(n_params,))
            for idx, group in enumerate(model):
                if points[idx] < self.prob:
                    offs1[group], offs2[group] = offs2[group].copy(), offs1[group]

            offs.append(offs1)
            offs.append(offs2)
        
        return np.reshape(offs, ga.pop.shape)
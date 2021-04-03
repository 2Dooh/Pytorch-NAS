import computational_graphs.operators.base as base

import numpy as np

class GOM(base.OperatorBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.trial_pop = None
        self.model = None
    
    def __call__(self, pop, f_pop, model, **kwargs):
        self.trial_pop = pop.copy()
        self.model = model
        
        N = pop.shape[0]
        n_evals = 0
        random_indices = np.arange(N)
        np.random.shuffle(random_indices)
        for i in range(N):
            np.random.shuffle(self.model)
            for group in self.model:
                d = self.trial_pop[random_indices[i]]
                x_dash = self.trial_pop[i].copy()
                x_dash[group] = d[group]
                y_dash = self.problem._f(x_dash)
                n_evals += 1
                if self.problem._compare(y_dash, f_pop[i]):
                    self.trial_pop[i] = x_dash
        return self.trial_pop, n_evals
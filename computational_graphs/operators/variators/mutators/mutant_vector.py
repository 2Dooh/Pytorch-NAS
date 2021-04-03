import computational_graphs.operators.base as base

import numpy as np
from numpy.random import uniform

class MutantVector(base.OperatorBase):
    def __init__(self, f, **kwargs):
        super().__init__(**kwargs)
        self.f = f
        self.pop_idx = None
        self.pop = None
    
    def create_mutant_vector(self, idx):
        x_r = np.random.choice(self.pop_idx[self.pop_idx != idx], 3)
        return self.pop[x_r[0]] + self.f * (self.pop[x_r[1]] - self.pop[x_r[2]])


    def __call__(self, pop):
        self.pop = pop
        self.pop_idx = np.arange(len(pop))
        mutant_vectors = np.array(list(map(self.create_mutant_vector, self.pop_idx)))
        return mutant_vectors
                            
        
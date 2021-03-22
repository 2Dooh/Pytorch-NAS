from typing import Tuple

import computational_graphs.operators.base as base

import torch

import numpy as np

from utils.evo_alg.denormalization import denormalize

class RandomSampling(base.OperatorBase):
    def __init__(self, size: int, **kwargs):
        super().__init__(**kwargs)
        self.size = size

    def __call__(self):
        pop = \
            np.random.rand(
                self.size, 
                self.problem.n_params, 
            ).astype(self.problem.type)
        xl, xu = self.problem.domain
        pop = denormalize(pop, xl, xu)
        return pop
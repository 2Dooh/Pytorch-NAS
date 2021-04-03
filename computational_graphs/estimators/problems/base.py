from abc import ABC, abstractmethod

import logging

# from typing import final

import torch

import numpy as np

class ProblemBase(ABC):
    def __init__(self,
                n_params,
                type,
                n_obj,
                constraints=None,
                vectorized=True,
                **kwargs):
        self.logger = logging.getLogger(name=self.__class__.__name__)
        if n_params <= 0:
            self.logger.error('Parameters length must be greater than zero')
        self.n_params = n_params
        self.n_obj = n_obj
        self.constraints = constraints
        self.domain = None
        self.type = type
        self.vectorized = vectorized
        self.argopt = None
        self.indices_dict = {}
        # self.comparer = self._compare


    # @final
    def eval(self, pop, **kwargs):
        # f_pop = torch.vmap(
        #     self._f, 
        #     in_dims=pop.size(), 
        #     out_dims=torch.Size([pop.size(0), -1])
        # )(pop)
        if self.vectorized:
          f_pop = self._f(pop)
        else:
          f_pop = []
          for i in range(pop.shape[0]):
            x = pop[i]
            f_x = self._f(x)
            f_pop += [f_x]
          f_pop = np.reshape(f_pop, (pop.shape[0], -1))
        return f_pop

    @abstractmethod
    def _f(self, **kwargs):
        raise NotImplementedError
    @abstractmethod
    def _compare(**kwargs):
        raise NotImplementedError


    

    
        
from abc import ABC, abstractmethod

import logging

from typing import final

import torch

class ProblemBase(ABC):
    def __init__(self,
                n_params,
                type,
                n_obj,
                constraints=None,
                **kwargs):
        self.logger = logging.getLogger(name=self.__class__.__name__)
        if n_params <= 0:
            self.logger.error('Parameters length must be greater than zero')
        self.n_params = n_params
        self.n_obj = n_obj
        self.constraints = constraints
        self.domain = None
        self.type = type

        self.argopt = None
        # self.comparer = self._compare


    @final
    def eval(self, pop, **kwargs):
        # f_pop = torch.vmap(
        #     self._f, 
        #     in_dims=pop.size(), 
        #     out_dims=torch.Size([pop.size(0), -1])
        # )(pop)
        f_pop = self._f(pop)
        return f_pop

    @abstractmethod
    def _f(self, **kwargs):
        raise NotImplementedError
    @abstractmethod
    def _compare(**kwargs):
        raise NotImplementedError


    

    
        
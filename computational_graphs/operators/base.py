from abc import ABC

import logging

class OperatorBase(ABC):
    def __init__(self, problem, **kwargs):
        self.problem = problem
        self.logger = logging.getLogger(name=self.__class__.__name__)

    def __call__(self, **kwargs):
        if 'pop' in kwargs.keys():
            return kwargs['pop']
        else:
            raise NotImplementedError
from abc import ABC, abstractmethod

import logging

class OperatorBase(ABC):
    def __init__(self, problem, **kwargs):
        self.problem = problem
        self.logger = logging.getLogger(name=self.__class__.__name__)

    @abstractmethod
    def __call__(self, **kwargs):
        raise NotImplementedError
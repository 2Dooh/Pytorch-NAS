
from pymoo.util.nds.non_dominated_sorting import find_non_dominated

import numpy as np

import logging

class ElitistArchive:
    def __init__(self, verbose=True) -> None:
        self.archive = {}
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        

    def __acceptance_test(self, f, key):
        if len(self.archive) == 0:
            return True 
        elif key not in self.archive['keys'] and\
            len(find_non_dominated(f, self.archive['F'])) > 0:
            return True
        else:
            return False


    def insert(self, x, f, key):
        if self.__acceptance_test(f, key):
            if len(self.archive) == 0:
                self.archive.update({
                    'X': x,
                    'F': f,
                    'keys': [key]
                })
            else:
                keys = np.row_stack([self.archive['keys'], key])
                X = np.row_stack([self.archive['X'], x])
                F = np.row_stack([self.archive['F'], f])
                I = find_non_dominated(F, F)

                self.archive.update({
                    'X': X[I],
                    'F': F[I],
                    'keys': keys[I].tolist()
                })
            if self.verbose:
                self.logger.info('Archive candidate with fitness: {}'.format(f))
                self.logger.info('Current archive size: {}'.format(len(self.archive['F'])))
            return True
        return False

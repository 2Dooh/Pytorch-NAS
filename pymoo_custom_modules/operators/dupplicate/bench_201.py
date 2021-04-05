from pymoo.model.duplicate import ElementwiseDuplicateElimination

import numpy as np

import logging

class Bench201DuplicateEliminator(ElementwiseDuplicateElimination):
    def __init__(self, **kwargs) -> None:
        super().__init__(cmp_func=self.is_equal, **kwargs)

    def is_equal(self, a, b):
        x, y = self.__decode(a.get('X')), self.__decode(b.get('X'))
        result = (x == y).all()
        if result:
            logging.info('dupplicate: - {} = {}'.format(x, y))
        return result

    @staticmethod
    def __decode(x):
        b2i = lambda a: int(''.join(str(int(bit)) for bit in a), 2)
        y = [b2i(x[start:start+3]) for start in np.arange(x.shape[0])[::3]]
        return np.array(y)
from pymoo.model.duplicate import ElementwiseDuplicateElimination

import numpy as np

import logging

from computational_graphs.bench101.nasbench.nasbench.lib.graph_util import is_full_dag, is_isomorphic, gen_is_edge_fn

INPUT = -1
OUTPUT = -2
CONV3X3 = 1
CONV1X1 = 2
MAXPOOL3X3 = 3
N_BITS_CONNECTIONS = 21
NUM_VERTICES = 7
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
BIT_PER_OP = 2
class Bench101DuplicateEliminator(ElementwiseDuplicateElimination):
    def __init__(self, **kwargs) -> None:
        super().__init__(cmp_func=self.is_equal, **kwargs)

    def is_equal(self, a, b):
        x, y = self.__decode(a.get('X')), self.__decode(b.get('X'))
        result = is_isomorphic(x, y)
        if result:
            logging.info('dupplicate: - {} = {}'.format(x, y))
        return result 

    @staticmethod
    def __decode(x):
        dag, ops = np.split(x, [N_BITS_CONNECTIONS])

        matrix = np.zeros((NUM_VERTICES, NUM_VERTICES))
        iu = np.triu_indices(NUM_VERTICES, 1)
        matrix[iu] = dag

        b2i = \
            lambda a: int(''.join(str(int(bit)) for bit in a), 2)
        ops_indices = \
            [b2i(ops[start:start+BIT_PER_OP]) for start in np.arange(ops.shape[0])[::BIT_PER_OP]]
        ops = np.array(ALLOWED_OPS)[ops_indices].tolist()

        return matrix.astype(np.int), [INPUT] + ops + [OUTPUT]
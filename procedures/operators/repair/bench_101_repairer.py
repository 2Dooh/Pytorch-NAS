import numpy as np

from pymoo.model.repair import Repair

from lib.nasbench101.nasbench import api
from lib.nasbench101.nasbench.lib.graph_util import num_edges, is_full_dag, gen_is_edge_fn

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix
N_BITS_CONNECTIONS = 21
N_BITS_OPERATIONS = 10
BIT_PER_OP = 2
class nasbench101Repairer(Repair):
    def _do(self, problem, pop, **kwargs):
        Z = pop.get('X')
        for i in range(Z.shape[0]):
            x_edge = Z[i][:N_BITS_CONNECTIONS]
            x_ops = Z[i][N_BITS_CONNECTIONS:]
            for start in np.arange(x_ops.shape[0])[::BIT_PER_OP]:
                y = x_ops[start:start+BIT_PER_OP]
                if y[0] == 1 and y.sum() > 1:
                    Z[i][N_BITS_CONNECTIONS+start] = 0

            matrix = np.zeros((NUM_VERTICES, NUM_VERTICES))
            iu = np.triu_indices(NUM_VERTICES, 1)
            matrix[iu] = x_edge
            b2i = \
                lambda a: int(''.join(str(int(bit)) for bit in a), 2)
            ops_indices = \
                [b2i(x_ops[start:start+BIT_PER_OP]) for start in np.arange(x_ops.shape[0])[::BIT_PER_OP]]
            ops = np.array(ALLOWED_OPS)[ops_indices].tolist()
            ops = [INPUT] + ops + [OUTPUT]
            spec = api.ModelSpec(
                matrix=matrix,
                ops=ops
            )
            while not problem.api.is_valid(spec):
                x_edge = (np.random.rand(N_BITS_CONNECTIONS) < 0.5).astype(np.bool)
                matrix = np.zeros((NUM_VERTICES, NUM_VERTICES))
                matrix[iu] = x_edge
                spec = api.ModelSpec(
                    matrix=matrix,
                    ops=ops
                )
            Z[i][:N_BITS_CONNECTIONS] = x_edge


        pop.set('X', Z)
        return pop
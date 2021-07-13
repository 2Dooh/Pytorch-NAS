import random

import copy

from pymoo.model.repair import Repair

from lib.benchmarks.nasbench101.nasbench import api

import numpy as np

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
class Bench101Repair(Repair):
    def _do(self, problem, pop, **kwargs):
        Z = pop.get('X')
        for i in range(Z.shape[0]):
            x_edge = Z[i][:N_BITS_CONNECTIONS]
            x_ops = Z[i][N_BITS_CONNECTIONS:]

            matrix = np.zeros((NUM_VERTICES, NUM_VERTICES))
            iu = np.triu_indices(NUM_VERTICES, 1)
            matrix[iu] = x_edge
            ops = np.array(ALLOWED_OPS)[x_ops.astype(np.int)].tolist()
            ops = [INPUT] + ops + [OUTPUT]
            spec = api.ModelSpec(
                matrix=matrix,
                ops=ops
            )
            if not problem.nasbench.is_valid(spec):
                matrix = self.mutate_spec(problem.nasbench, spec)
                # np.random.shuffle(x_edge)
                # matrix = np.zeros((NUM_VERTICES, NUM_VERTICES))
                # matrix[iu] = x_edge
                spec = api.ModelSpec(
                    matrix=matrix,
                    ops=ops
                )
            Z[i][:N_BITS_CONNECTIONS] = matrix[iu]


        pop.set('X', Z)
        return pop

    @staticmethod
    def mutate_spec(nasbench, old_spec, mutation_rate=1.0):
        """Computes a valid mutated spec from the old_spec."""
        while True:
            new_matrix = copy.deepcopy(old_spec.original_matrix)
            new_ops = copy.deepcopy(old_spec.original_ops)

            # In expectation, V edges flipped (note that most end up being pruned).
            edge_mutation_prob = mutation_rate / NUM_VERTICES
            for src in range(0, NUM_VERTICES - 1):
                for dst in range(src + 1, NUM_VERTICES):
                    if random.random() < edge_mutation_prob:
                        new_matrix[src, dst] = 1 - new_matrix[src, dst]
                
            # # In expectation, one op is resampled.
            # op_mutation_prob = mutation_rate / OP_SPOTS
            # for ind in range(1, NUM_VERTICES - 1):
            #     if random.random() < op_mutation_prob:
            #         available = [o for o in api.config['available_ops'] if o != new_ops[ind]]
            #         new_ops[ind] = random.choice(available)
                
            new_spec = api.ModelSpec(new_matrix, new_ops)
            if nasbench.is_valid(new_spec):
                return new_matrix
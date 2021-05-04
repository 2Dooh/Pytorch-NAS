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

def _decode(x):
        dag, ops = np.split(x, [21])

        matrix = np.zeros((NUM_VERTICES, NUM_VERTICES))
        iu = np.triu_indices(NUM_VERTICES, 1)
        matrix[iu] = dag

        b2i = lambda a: int(''.join(str(int(bit)) for bit in a), 2)
        ops_indices = [b2i(ops[start:start+2]) for start in np.arange(ops.shape[0])[::2]]
        ops = np.array(ALLOWED_OPS)[ops_indices].tolist()

        return matrix, [INPUT] + ops + [OUTPUT]
np.random.seed(0)
x = np.random.randint(0, 2, (31))

res =_decode(x)
print(res)
print(len(res))
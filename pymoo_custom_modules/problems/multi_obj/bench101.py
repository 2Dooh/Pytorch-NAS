# from pymoo.model.problem import Problem

# import numpy as np

# from computational_graphs.api.nasbench.nasbench import api
# from computational_graphs.api.nasbench.nasbench.lib.model_builder import build_model_fn, build_module

# import os

# import tensorflow.compat.v1 as tf

# INPUT = 'input'
# OUTPUT = 'output'
# CONV3X3 = 'conv3x3-bn-relu'
# CONV1X1 = 'conv1x1-bn-relu'
# MAXPOOL3X3 = 'maxpool3x3'
# NUM_VERTICES = 7
# MAX_EDGES = 9
# EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
# OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
# ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
# ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix
# N_BITS_CONNECTIONS = 21
# N_BITS_OPERATIONS = 10
# BIT_PER_OP = 2

# NAS_BENCH = api.NASBench(os.path.join(os.environ('TORCH_HOME'), 'nasbench_only108.tfrecord'))

# class Bench101(Problem):
#     def __init__(self,
#                  dataset,
#                  n_var, 
#                  n_obj, 
#                  n_constr, 
#                  xl, 
#                  xu, 
#                  type_var, 
#                  evaluation_of, 
#                  replace_nan_values_of, 
#                  parallelization, 
#                  elementwise_evaluation, 
#                  exclude_from_serialization, 
#                  callback,
#                  epochs=12,
#                  **kwargs):
#         super().__init__(n_var=N_BITS_CONNECTIONS+N_BITS_OPERATIONS, 
#                          n_obj=2, 
#                          n_constr=0, 
#                          xl=np.zeros(N_BITS_CONNECTIONS+N_BITS_OPERATIONS), 
#                          xu=np.ones(N_BITS_CONNECTIONS+N_BITS_OPERATIONS),  
#                          elementwise_evaluation=True, 
#                          **kwargs)

#         self.epochs = epochs
#         self.nasbench = api.NASBench(os.path.join(os.environ('TORCH_HOME'), 'nasbench_only108.tfrecord'))

#     def _evaluate(self, x, out, *args, **kwargs):
#         matrix, ops = self._decode(x)
#         cell = api.ModelSpec(
#             matrix=matrix,
#             ops=ops
#         )
#         assert(NAS_BENCH.is_valid(cell))
#         data = NAS_BENCH.query(cell, epochs=self.epochs)
#         net = build_module(cell, self.inputs, self.channels, is_training=self.is_training)

#         return super()._evaluate(x, out, *args, **kwargs)

#     def _decode(x):
#         dag, ops = np.split(x, [N_BITS_CONNECTIONS])

#         matrix = np.zeros((NUM_VERTICES, NUM_VERTICES))
#         iu = np.triu_indices(NUM_VERTICES, 1)
#         matrix[iu] = dag

#         b2i = \
#             lambda a: int(''.join(str(int(bit)) for bit in a), 2)
#         ops_indices = \
#             [b2i(ops[start:start+2]) for start in np.arange(ops.shape[0])[::BIT_PER_OP]]
#         ops = np.array(ALLOWED_OPS)[ops_indices].tolist()

#         return matrix, [INPUT] + ops + [OUTPUT]


#     def _construct_problem(self):
#         pass

#     def get_flops(model):
#         session = tf.compat.v1.Session()
#         graph = tf.compat.v1.get_default_graph()

#         with graph.as_default():
#             run_meta = tf.compat.v1.RunMetadata()
            

        
        
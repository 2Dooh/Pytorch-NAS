from pymoo.model.problem import Problem

import numpy as np

from utils.neural_net.flops_benchmark import get_model_infos
from utils.neural_net.gf_metric import GradientFreeEvaluator

from lib.bench101.nasbench.nasbench import api
from lib.bench101.nasbench.nasbench.lib.graph_util import gen_is_edge_fn
from lib.bench101.nasbench_pytorch.model import Network

import os

import logging

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

# NAS_BENCH = api.NASBench(os.path.join(os.environ('TORCH_HOME'), 'nasbench_only108.tfrecord'))

class Bench101(Problem):
    CIFAR10_SHAPE = [1, 3, 32, 32]
    CIFAR10_N_LABELS = 10
    def __init__(self,
                 epoch=12,
                 obj_list=['flops', 'validation'],
                 **kwargs):
        super().__init__(n_var=N_BITS_CONNECTIONS+N_BITS_OPERATIONS, 
                         xl=np.zeros(N_BITS_CONNECTIONS+N_BITS_OPERATIONS), 
                         xu=np.ones(N_BITS_CONNECTIONS+N_BITS_OPERATIONS),   
                         **kwargs)
        self.obj_list = obj_list
        self.logger = logging.getLogger(name=self.__class__.__name__)
        self.score_dict = {}
        self.epochs = epoch
        self.arch_info = \
            'idx: {} - ' + \
            ' - '.join('{}: '.format(f.replace('accuracy', 'error')) + '{}' for f in self.obj_list)
        # self.api = api.NASBench(os.path.join('~', 'nasbench_only108.tfrecord'))
        self.api = api.NASBench('experiments/nasbench_full.tfrecord')
        #os.environ['TORCH_HOME']

    def _evaluate(self, x, out, *args, **kwargs):
        matrix, ops, str_x = self._decode(x)
        if str_x in self.score_dict:
            F = self.score_dict[str_x]
            out['F'] = np.column_stack(F)
            self.logger.info('Re-evaluated arch: {}'.format(str_x))
            self.logger.info(self.arch_info.format(str_x, *F))
            return
        spec = api.ModelSpec(
            matrix=matrix,
            ops=ops
        )
        # assert(NAS_BENCH.is_valid(cell))
        err = (1 - self.api.query(spec, epochs=self.epochs)['{}_accuracy'.format(self.obj_list[1])]) * 100
        net = Network(spec, self.CIFAR10_N_LABELS)
        flops, params = get_model_infos(net, self.CIFAR10_SHAPE)
        complexity = {'flops': flops, 'n_params': params}
        F = [complexity[self.obj_list[0]], err]
        self.logger.info(self.arch_info.format(str_x, *F))
        out['F'] = np.column_stack(F)

    @staticmethod
    def _decode(x):
        str_x = ''.join(str(int(bit)) for bit in x)
        dag, ops = np.split(x, [N_BITS_CONNECTIONS])

        matrix = np.zeros((NUM_VERTICES, NUM_VERTICES))
        iu = np.triu_indices(NUM_VERTICES, 1)
        matrix[iu] = dag

        b2i = \
            lambda a: int(''.join(str(int(bit)) for bit in a), 2)
        ops_indices = \
            [b2i(ops[start:start+BIT_PER_OP]) for start in np.arange(ops.shape[0])[::BIT_PER_OP]]
        ops = np.array(ALLOWED_OPS)[ops_indices].tolist()

        return matrix.astype(np.int), [INPUT] + ops + [OUTPUT], str_x


    def _construct_problem(self):
        pass
            

        
        
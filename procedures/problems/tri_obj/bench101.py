from lib.benchmarks.nasbench101.nasbench import api

from lib.nasbench_pytorch.model import Network

import numpy as np

import logging

from utils.moeas.elitist_archive import ElitistArchive

from utils.neural_net.flops_benchmark import get_model_infos
from utils.neural_net.gf_metric import GradientFreeEvaluator

from pymoo.model.problem import Problem

import os
from os.path import expanduser

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]


class Bench101(Problem):
    EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) // 2   # Upper triangular matrix
    OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
    def __init__(self,
                 benchmark_path,
                 net_config,
                 epoch=12,
                 load_benchmark=True,
                 **kwargs):
        edge_ub = np.ones(self.EDGE_SPOTS)
        edge_lwb = np.zeros(self.EDGE_SPOTS)
        op_ub = np.ones(self.OP_SPOTS) * max(range(len(ALLOWED_OPS)))
        op_lwb = np.zeros(self.OP_SPOTS)
        super().__init__(n_var=self.EDGE_SPOTS+self.OP_SPOTS, 
                         xl=np.concatenate([edge_lwb, op_lwb]), 
                         xu=np.concatenate([edge_ub, op_ub]),   
                         **kwargs)
        self.net_config = net_config
        self.logger = logging.getLogger(name=self.__class__.__name__)
        self.score_dict = {}
        self.epochs = epoch
        self.nasbench = None
        if load_benchmark and '~' in benchmark_path:
            self.nasbench = api.NASBench(os.path.join(expanduser('~'), benchmark_path.replace('~/', '')))
        elif load_benchmark:
            self.nasbench = api.NASBench(benchmark_path)
        self.elitist_archive = ElitistArchive(filter_duplicate_by_key=False)

    def _evaluate(self, x, out, *args, **kwargs):
        dag, ops = np.split(x, [self.EDGE_SPOTS])
        key = (tuple(dag.tolist()), tuple(ops.tolist()))
        if key in self.score_dict:
            out['F'] = self.score_dict[key]
            self.logger.info('Re-evaluated arch: {}'.format(key))
            return
        
        matrix, ops = self._decode(dag, ops)
        spec = api.ModelSpec(
            matrix=matrix,
            ops=ops
        )

        err = (1 - self.nasbench.query(spec, epochs=self.epochs)['validation_accuracy']) * 100
        network = Network(spec, self.net_config.num_labels)
        flops, _ = get_model_infos(network, self.net_config.input_size)
        F = [flops, err]
        out['F'] = np.column_stack(F)
        self.score_dict[key] = out['F']
        self.elitist_archive.insert(x, out['F'], key=key)

    @staticmethod
    def _decode(dag, ops):
        matrix = np.zeros((NUM_VERTICES, NUM_VERTICES))
        iu = np.triu_indices(NUM_VERTICES, 1)
        matrix[iu] = dag

        ops = np.array(ALLOWED_OPS)[ops.astype(np.int)].tolist()

        return matrix.astype(np.int), [INPUT] + ops + [OUTPUT]

class Bench101GradientFree(Bench101):
    def __init__(self, evaluator_config, **kwargs):
        super().__init__(n_obj=3, load_benchmark=True, **kwargs)
        self.evaluator = GradientFreeEvaluator(**evaluator_config)

    def _evaluate(self, x, out, *args, **kwargs):
        dag, ops = np.split(x, [self.EDGE_SPOTS])
        key = (tuple(dag.tolist()), tuple(ops.tolist()))
        if key in self.score_dict:
            out['F'] = self.score_dict[key]
            self.logger.info('Re-evaluated arch: {}'.format(key))
            return
        matrix, ops = self._decode(dag, ops)
        spec = api.ModelSpec(
            matrix=matrix,
            ops=ops
        )
        assert(self.nasbench.is_valid(spec))
        network_thin = Network(spec, 
                              in_channels=1, 
                              num_labels=self.net_config.num_labels,
                              stem_out_channels=np.sum(matrix[1:], axis=0)[-1],
                              num_stack=self.net_config.num_stack,
                              num_modules_per_stack=self.net_config.num_modules_per_stack,
                              use_stem=False).cuda()
        
        lrs = self.evaluator.calc_lrc(network_thin)
        network = Network(spec, self.net_config.num_labels).cuda()
        flops, _ = get_model_infos(network, self.net_config.input_size)
        ntks = self.evaluator.calc_ntk(network)
        

        ntk = np.log(max(ntks))
        lr = min(lrs)

        F = [flops, ntk, -lr]
        out['F'] = np.column_stack(F)
        self.score_dict[key] = out['F']
        self.elitist_archive.insert(x, out['F'], key=key)

            

        
        
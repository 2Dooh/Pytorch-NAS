from pymoo.model.problem import Problem
from pymoo.util.normalization import denormalize

from utils.moeas.elitist_archive import ElitistArchive
from utils.neural_net.gf_metric import GradientFreeEvaluator

from lib.customs.mlp import MultiLayerPerceptron

import numpy as np

import logging

class DynamicMLP(Problem):
    def __init__(self, 
                 input_size,
                 output_size,
                 node_bound,
                 max_depth,
                 evaluator_cfg,
                 **kwargs):
        self.max_depth = max_depth
        n_var = 3 * max_depth

        super().__init__(
            n_var=n_var, 
            n_obj=2, 
            xl=np.zeros(n_var), 
            xu=np.ones(n_var), 
            elementwise_evaluation=True
        )
        self.input_size = input_size
        self.output_size = output_size
        self.node_bound = node_bound
        self.max_depth = max_depth
        self.archive = ElitistArchive(verbose=True)
        self.fitness_dict = {}
        self.evaluator_cfg = evaluator_cfg
        self.evaluator = GradientFreeEvaluator(**evaluator_cfg)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _evaluate(self, x, out, *args, **kwargs):
        x = np.split(x, np.arange(len(x))[::3])[1:]
        assert len(x) == self.max_depth
        backbones = [
            (int(denormalize(n_nodes, *self.node_bound)), drop_prob) \
                for n_nodes, drop_prob, layer_prob in x if layer_prob > .5
        ]


        #backbones = [(int(denormalize(n_nodes, 32, 1024)), drop_prob) for n_nodes, drop_prob, layer_prob in x if layer_prob > .5]
        
        key = tuple([
            (int(denormalize(n_nodes, *self.node_bound)), drop_prob, layer_prob > .5) \
                for n_nodes, drop_prob, layer_prob in x
        ])
        if key in self.fitness_dict:
            out['F'] = self.fitness_dict[key]
            return
        x_min, x_max = self.node_bound
        backbones_thin = [
            (int(denormalize(n_nodes, 1, x_max/x_min)), drop_prob) \
                for n_nodes, drop_prob, _ in x
        ]
        backbones_thin = [(np.product(self.evaluator_cfg['lr_input_size'][1:]), None)] + backbones_thin
        backbones = [(self.input_size, None)] + backbones
        net = MultiLayerPerceptron(self.output_size, backbones).cuda()
        net_thin = MultiLayerPerceptron(self.output_size, backbones_thin).cuda()
        ntks = self.evaluator.calc_ntk(net)
        ntk = max(ntks)
        lrs = self.evaluator.calc_lrc(net_thin)
        lr = min(lrs)
        out['F'] = np.column_stack([ntk, lr])
        self.logger.info('arch: {} - ntk: {} - lr: {}'.format(key, ntk, lr))
        self.fitness_dict[key] = out['F']
        self.archive.insert(x, out['F'], key=np.array(key).ravel().tolist())
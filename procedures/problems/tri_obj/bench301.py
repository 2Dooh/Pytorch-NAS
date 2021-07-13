from lib.darts.encoding import Genotype, decode, convert
from lib.darts.cnn.model import NetworkCIFAR

import lib.benchmarks.nasbench301.nasbench301 as nb

from pymoo.model.problem import Problem

import numpy as np

from easydict import EasyDict as edict

import logging

from utils.moeas.elitist_archive import ElitistArchive

from utils.neural_net.flops_benchmark import get_model_infos
from utils.neural_net.gf_metric import GradientFreeEvaluator

class DARTSSurrogateModel(Problem):
    N_CELLS = 2 # normal + reduce
    N_INPUTS = 2
    STRATEGIES = {
        'rand': np.random.choice, 
        'avg': np.mean, 
        'best': max, 
        'worst': min
    }
    def __init__(self,
                 net_config,
                 surrogate_model_paths: list,  
                 strategy='random', 
                 use_noise=False,              
                 **kwargs):
        self.net_config = edict(net_config)

        n_var = int(4 * self.net_config.n_blocks * self.N_CELLS)
        xl, xu = self.__get_bound(n_var)

        super().__init__(n_var=n_var, 
                         xl=xl, 
                         xu=xu, 
                         **kwargs)

        assert(strategy in self.STRATEGIES.keys())
        self.strategy = strategy
        self.use_noise = use_noise

        self.score_dict = {}

        self.evaluators = []
        for path in surrogate_model_paths:
            self.evaluators += [nb.load_ensemble(path)]

        self.logger = logging.getLogger(self.__class__.__name__)

        self.elitist_archive = ElitistArchive(verbose=True)

    def __get_bound(self, n_var):
        xu = np.ones(n_var) * max(range(self.net_config.n_ops))
        edge_ub = []
        for n in range(self.net_config.n_blocks):
            edge_ub += [max(range(self.N_INPUTS+n))]
        edge_ub = np.repeat(edge_ub, repeats=self.N_INPUTS)
        # for normal cell and reduce celledge_ub[::2] -= 1
        # prevent 2nd input to cell the same as the 1st one
        xu[n_var//2:] = xu[:n_var//2]
        xl = np.zeros_like(xu)
        return xl, xu


    def _evaluate(self, x, out, *args, **kwargs):
        key = tuple(x.tolist())
        if key in self.score_dict:
            self.logger.info('Re-evaluated arch: {}'.format(key))
            out['F'] = self.score_dict[key]
            return

        genotype = decode(convert(x))
        normal = genotype.normal; reduce = genotype.reduce
        normal_concat = list(range(2, 6)); reduce_concat = list(range(2, 6))
        genotype = Genotype(normal=normal,
                            reduce=reduce,
                            normal_concat=normal_concat,
                            reduce_concat=reduce_concat)

        network = NetworkCIFAR(genotype=genotype, **self.net_config).cuda()
        network.drop_path_prob = -1
        flops, _ = get_model_infos(network, self.net_config.input_size)

        scores = [
            predictor.predict(
                genotype, 
                representation='genotype', 
                with_noise=self.use_noise
            ) for predictor in self.evaluators
        ]

        score = self.STRATEGIES[self.strategy](scores)
        err = 100 - score

        F = [flops, err]
        out['F'] = np.column_stack(F)

        self.score_dict[key] = out['F']
        self.elitist_archive.insert(x, out['F'], key=key)
    

class DARTSGradientFree(DARTSSurrogateModel):
    def __init__(self, evaluator_config, **kwargs):
        super().__init__(n_obj=3,
                         surrogate_model_paths=[], 
                         **kwargs)
        self.evaluator = GradientFreeEvaluator(**evaluator_config)
            

    def _evaluate(self, x, out, *args, **kwargs):
        key = tuple(x.tolist())
        if key in self.score_dict:
            self.logger.info('Re-evaluated arch: {}'.format(key))
            out['F'] = self.score_dict[key]
            return
        genotype = decode(convert(x))
        normal = genotype.normal; reduce = genotype.reduce
        normal_concat = list(range(2, 6)); reduce_concat = list(range(2, 6))
        genotype = Genotype(
            normal=normal,
            reduce=reduce,
            normal_concat=normal_concat,
            reduce_concat=reduce_concat
        )

        network = NetworkCIFAR(genotype=genotype, **self.net_config).cuda()
        network.drop_path_prob = -1

        network_thin = NetworkCIFAR(C=1,
                                    num_classes=self.net_config.num_classes,
                                    layers=self.net_config.layers,
                                    auxiliary=self.net_config.auxiliary,
                                    genotype=genotype,
                                    use_stem=False).cuda()
        network_thin.drop_path_prob = -1
        flops, _ = get_model_infos(network, self.net_config.input_size)
        lrs = self.evaluator.calc_lrc(network_thin)
        ntks = self.evaluator.calc_ntk(network)

        ntk = np.log(max(ntks))
        lr = min(lrs)

        F = [flops, ntk, -lr]
        out['F'] = np.column_stack(F)
        self.score_dict[key] = out['F']

        self.elitist_archive.insert(x, out['F'], key=key)
            
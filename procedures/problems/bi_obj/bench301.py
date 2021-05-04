from pymoo.model.problem import Problem

from utils.neural_net.gf_metric import GradientFreeEvaluator
from utils.neural_net.flops_benchmark import get_model_infos

from utils.moeas.elitist_archive import ElitistArchive

from lib.darts.encoding import Genotype, decode, convert
from lib.darts.cnn.model import NetworkCIFAR

import numpy as np

import lib.benchmarks.nasbench301.nasbench301 as nb

import logging

BIT_PER_OP = 3


class Darts(Problem):
    INIT_CHANNELS = 32
    LAYERS = 8
    AUXILIARY = True
    def __init__(self,
                 n_blocks=4,
                 n_ops=7,
                 xgb=True,
                 lgb=True,
                 num_classes=10,
                 input_size=[1, 3, 32, 32],
                 obj_list=['flops', 'test_error'],
                 **kwargs):
        
        n_var = int(4 * n_blocks * 2)
        xu = np.ones(n_var) * (n_ops-1)
        edge_ub = []
        for n in range(n_blocks):
            edge_ub += [len(list(range(2+n))) - 1]
        
        edge_ub = np.repeat(edge_ub, repeats=2) # for normal cell and reduce cell
        edge_ub[::2] -= 1 # prevent 2nd input to cell the same as the 1st one
        xu[1:n_var//2:2] = edge_ub
        xu[n_var//2:] = xu[:n_var//2]

        xl = np.zeros_like(xu)

        super().__init__(n_var=n_var, 
                         xl=xl, 
                         xu=xu, 
                         **kwargs)
        self.input_size = input_size
        self.num_classes = num_classes
        self.obj_list = obj_list

        self.b2i_func = lambda a: int(''.join(str(int(bit)) for bit in a), 2)
        self.score_dict = {}
        self.arch_info = \
            'idx: {} - ' + \
            ' - '.join('{}: '.format(f.replace('accuracy', 'error')) + '{}' for f in self.obj_list)

        self.xgb, self.lgb = None, None
        if xgb:
            path = 'experiments/nasbench301_models_v1.0/nb_models/xgb_v1.0'
            self.xgb = nb.load_ensemble(path)
        if lgb:
            path = 'experiments/nasbench301_models_v1.0/nb_models/lgb_runtime_v1.0'
            self.lgb = nb.load_ensemble(path)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.elitist_archive = ElitistArchive(verbose=True)

    
    @staticmethod
    def __bin2int(func, x):
        x_int = [func(x[start:start+BIT_PER_OP]) for start in np.arange(x.shape[0])[::BIT_PER_OP]]
        x_str = ''.join(str(x_i) for x_i in x_int)
        return x_int, x_str

    def _evaluate(self, x, out, *args, **kwargs):

        x_str = ''.join(str(x_i) for x_i in x)
        genotype = decode(convert(x))
        normal = genotype.normal; reduce = genotype.reduce
        normal_concat = list(range(2, 6)); reduce_concat = list(range(2, 6))
        genotype = Genotype(normal=normal,
                            reduce=reduce,
                            normal_concat=normal_concat,
                            reduce_concat=reduce_concat)

        if x_str in self.score_dict:
            out['F'] = self.score_dict[x_str]
            self.logger.info('Re-evaluated arch: {}'.format(x_str))
            return

        model = NetworkCIFAR(C=32,
                             num_classes=10,
                             layers=8,
                             auxiliary=True,
                             genotype=genotype).cuda()
        model.drop_path_prob = 0.1
        flops, _ = get_model_infos(model, [1, 3, 32, 32])

        predicted_test_acc = self.xgb.predict(genotype, representation='genotype', with_noise=False)
    
        test_err = 100 - predicted_test_acc

        F = [flops, test_err]
        out['F'] = np.column_stack(F)

        self.score_dict[x_str] = out['F']
        self.logger.info(self.arch_info.format(x_str, *F))
        self.elitist_archive.insert(x, out['F'], x_str)
        # if self.elitist_archive['X'] is None:
        #     self.elitist_archive['X'] = x
        #     self.elitist_archive['F'] = out['F']
        #     self.elitist_archive['codes'] = np.array([x_str])
        # elif x_str not in self.elitist_archive['codes'].tolist():
        #     if len(find_non_dominated(out['F'], self.elitist_archive['F'])) > 0:

        #         codes = np.row_stack([self.elitist_archive['codes'], x_str])
        #         X_archive = np.row_stack([self.elitist_archive['X'], x])
        #         F_archive = np.row_stack([self.elitist_archive['F'], out['F']])
        #         # I = NonDominatedSorting().do(F_archive, only_non_dominated_front=True)
        #         I = find_non_dominated(F_archive, F_archive)

        #         self.elitist_archive['X'] = X_archive[I]
        #         self.elitist_archive['F'] = F_archive[I]
        #         self.elitist_archive['codes'] = codes[I]

        #         self.logger.info('Elitist archive size: {}'.format(len(self.elitist_archive['F'])))

import torch
from collections import namedtuple
from lib.models.cell_infers.nasnet_cifar import NASNetonCIFAR
class DartsGradientFree(Darts):
    def __init__(self, 
                 n_blocks=4, 
                 n_ops=7, 
                 cuda=True,
                 seed=0,
                 num_workers=4,
                 n_repeats=3,
                 dataset='cifar10',
                 **kwargs):
        super().__init__(n_blocks=n_blocks, 
                         n_ops=n_ops, 
                         xgb=False, 
                         lgb=False, 
                         n_obj=3,
                         obj_list=['flops', 'ntk', 'lr'], 
                         **kwargs)
        
        self.n_repeats = n_repeats
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.evaluator = GradientFreeEvaluator(dataset=dataset, seed=seed, num_workers=num_workers)

    @staticmethod
    def genotype_to_autodl_format(genotype):
        autodl_genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat connectN connects')

        cells = {'normal': None, 'reduce': None}
        for cell in ['normal', 'reduce']:
            lst = []
            blocks = eval('genotype.' + cell)
            for i in np.arange(len(blocks))[::2]:
                lst += [(blocks[i], blocks[i+1])]
            cells[cell] = lst

        new_genotype = autodl_genotype(normal=cells['normal'], 
                                        reduce=cells['reduce'], 
                                        normal_concat=genotype.normal_concat, 
                                        reduce_concat=genotype.reduce_concat,
                                        connectN=None,
                                        connects=None)

        return new_genotype
            

    def _evaluate(self, x, out, *args, **kwargs):
        x_str = ''.join(str(x_i) for x_i in x)
        genotype = decode(convert(x))
        normal = genotype.normal; reduce = genotype.reduce
        normal_concat = list(range(2, 6)); reduce_concat = list(range(2, 6))
        genotype = Genotype(
            normal=normal,
            reduce=reduce,
            normal_concat=normal_concat,
            reduce_concat=reduce_concat
        )

        if x_str in self.score_dict:
            self.logger.info('Re-evaluated arch: {}'.format(x_str))
        else:
            network = NetworkCIFAR(C=self.INIT_CHANNELS,
                                   num_classes=self.num_classes,
                                   layers=self.LAYERS,
                                   auxiliary=self.AUXILIARY,
                                   genotype=genotype).to(self.device)
            network.drop_path_prob = -1

            network_thin = NetworkCIFAR(C=1,
                                        num_classes=self.num_classes,
                                        layers=self.LAYERS,
                                        auxiliary=self.AUXILIARY,
                                        genotype=genotype,
                                        use_stem=False).to(self.device)
            network_thin.drop_path_prob = -1
            _flops = get_model_infos(network, self.input_size)
            _lrs = self.evaluator.calc_lrc(network_thin, n_repeats=self.n_repeats)
            _ntks = self.evaluator.calc_ntk(network, n_repeats=self.n_repeats)

            self.score_dict[x_str] = {'ntks': _ntks, 'lrs': _lrs, 'flops': _flops}

        _ntks = np.array(self.score_dict['x_str']['ntks'])
        _lrs = self.score_dict['x_str']['lrs']
        
        ntk = np.log(_ntks.max())
        lr = min(_lrs)
        flops = self.score_dict[x_str]['flops']

        F = [flops, ntk, lr]
        self.logger.info(self.arch_info.format(x_str, *F))

        out['F'] = np.column_stack(F)

        self.elitist_archive.insert(x, out['F'], x_str)
            
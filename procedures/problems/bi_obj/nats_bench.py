import numpy as np

import torch

from nats_bench import create

import lib.models.cell_operations as ops

from lib.models import get_cell_based_tiny_net

from easydict import EasyDict as edict

import os

from pymoo.model.problem import Problem


from utils.moeas.elitist_archive import ElitistArchive

import logging

from abc import abstractmethod

class NATSBench(Problem):
    def __init__(self, 
                search_space, 
                iepoch,
                dataset,
                hp='12', 
                trial_idx=0,
                **kwargs):

        super().__init__(**kwargs)
        self.trial_seed = trial_idx
        self.iepoch = iepoch
        self.hp = hp
        self.dataset = dataset
        self.api = create(None, search_space, fast_mode=True, verbose=False)

        self.logger = logging.getLogger(name=self.__class__.__name__)
        self.score_dict = {}
        self.elitist_archive = ElitistArchive()

    def _evaluate(self, x, out, *args, **kwargs):
        genotype = self._decode(x)
        index = self.api.query_index_by_arch(genotype)
        key = tuple(x.tolist())
        if key in self.score_dict:
            out['F'] = self.score_dict[key]
            self.logger.info('Re-evaluated arch: {}'.format(key))
            return
        
        flops = self.api.get_cost_info(index, self.dataset, self.hp)['flops']
        data = self.api.query_by_index(index, self.dataset, hp=self.hp)
        trial_seed = list(data.keys())[self.trial_idx]
        valid_acc = data[trial_seed].get_eval('valid', self.iepoch)['accuracy']
        valid_err = 100 - valid_acc
        F = [flops, valid_err]

        out['F'] = np.column_stack(F)
        self.score_dict[key] = out['F']
        
        self.elitist_archive.insert(x, out['F'], key=key)


    @abstractmethod
    def _decode(self, genotype):
        raise NotImplementedError

class SizeSearchSpace(NATSBench):
    CHANNELS = [8, 16, 24, 32, 40, 48, 56, 64]
    N_LAYERS = 5
    N_BIT_PER_CHANNEL = 3
    def __init__(self, **kwargs):
        super().__init__(n_var=self.N_LAYERS, 
                         xl=np.zeros(self.N_LAYERS),
                         xu=np.ones(self.N_LAYERS) * max(range(len(self.CHANNELS))),
                         search_space='sss', 
                         **kwargs)
        

    def _decode(self, x):
        channels = np.array(self.CHANNELS)[x]
        return ':'.join(str(channel) for channel in channels)


class TopologySearchSpace(NATSBench):
    MAX_NODES = 4
    def __init__(self, **kwargs):
        super().__init__(n_var=self.MAX_NODES*(self.MAX_NODES-1)//2, 
                         xl=np.zeros(self.MAX_NODES*(self.MAX_NODES-1)//2),
                         xu=np.ones(self.MAX_NODES*(self.MAX_NODES-1)//2) * max(range(len(ops.NAS_BENCH_201))),
                         search_space='tss', 
                         **kwargs)
        self.primitives = np.array(ops.NAS_BENCH_201)

    def _decode(self, x):
        # b2i = lambda a: int(''.join(str(int(bit)) for bit in a), 2)
        # decoded_indices = [b2i(genotype[start:start+self.n_bits_per_op]) for start in np.arange(genotype.shape[0])[::self.n_bits_per_op]]
        # ops = self.predefined_ops[decoded_indices]
        ops = self.primitives[x]
        node_str = '{}~{}'
        genotype = []
        k = 0
        for i in range(self.MAX_NODES):
            node_op = []
            for j in range(i):
                node_op += [node_str.format(ops[k], j)]
                k += 1
            if len(node_op) > 0:
                genotype += ['|' + '|'.join(node_op) + '|']
                
        genotype = '+'.join(genotype)
        return genotype

# import random
# from lib.custom_models.mlp import MultiLayerPerceptron
# class TSSSurrogateModel(TSSBench201):
#     NTK_MEAN = 11.109008924001119
#     NTK_STD = 1.1156648898147397
#     LR_MEAN = 753.2590219019432
#     LR_STD = 336.5909727826006
#     def __init__(self, 
#                 dataset='cifar10',
#                 input_size=(32, 3, 32, 32),
#                 cuda=True,
#                 seed=0, 
#                 num_workers=4,
#                 **kwargs):

#         super().__init__(n_obj=2,
#                          obj_list=['flops', 'pred_test-err'],
#                          dataset=dataset,
#                          elementwise_evaluation=True,
#                          **kwargs) 
#         self.device = torch.device('cuda:0' if cuda else 'cpu')
#         self.ntk_lr_metrics = GradientFreeEvaluator(input_size, dataset, 'best', seed, num_workers)

#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.enabled = True

#         self.predictor = MultiLayerPerceptron(
#             layers_dim=[2, 3, 3, 5, 1],
#             activations=["Tanh", "Tanh", "Tanh", ""]
#         )
#         state_dict = torch.load('experiments/NTK_LR-SurrogateModel/out/Regressor-NTK_LR_Regression.pth.tar')
#         self.predictor.load_state_dict(state_dict)


#     def _evaluate(self, x, out, algorithm, *args, **kwargs):
#         genotype = x
#         phenotype = self._decode(genotype)
#         index = self.api.query_index_by_arch(phenotype)
        
        
#         config = self.api.get_net_config(index, self.dataset)
#         config.update({
#             'C': 3,
#             'N': 1,
#             'depth': -1,
#             'C_in': 3,
#             'use_stem': True
#         })

#         config_thin = config.copy()
#         config_thin.update({
#             'C': 1,
#             'C_in': 1,
#             'N': 1,
#             'depth': 1,
#             'use_stem': True
#         })

#         network = get_cell_based_tiny_net(edict(config)).to(self.device)
#         network_thin = get_cell_based_tiny_net(edict(config_thin)).to(self.device)

#         cost_info = self.api.get_cost_info(index, self.dataset)
        

#         _flops = cost_info['flops']

#         if index in self.score_dict:    
#           self.logger.info('Re-evaluate arch: {}'.format(index))
#         else:
#           lrs = self.ntk_lr_metrics.calc_lrc(network_thin, n_repeats=3)
#           ntks = self.ntk_lr_metrics.calc_ntk(network, n_repeats=3)
          
#           self.score_dict[index] = {'ntks': ntks, 'lrs': lrs}
#           # self.score_dict[index] = {'ntks': ntks}
#         # ntk = (.7 * ntks.std()) + (.3 * ntks.max())
#         # ntk = np.abs(ntks.min()) if True in ntks < 0 else ntks.max()
#         ntks = np.nan_to_num(np.array(self.score_dict[index]['ntks']), 0, 1e9)
#         lrs = np.array(self.score_dict[index]['lrs'])
#         self.logger.info('NTK: {}'.format(np.clip(ntks, 0, 1e9).max()))
#         self.logger.info('LR: {}'.format(lrs.min()))
#         ntk = np.log(np.clip(ntks, 0, 1e7).mean())
#         lr = -lrs.mean()

#         standardized_ntk = (ntk - self.NTK_MEAN) / self.NTK_STD
#         standardized_lr = (lr - self.LR_MEAN) / self.LR_STD

#         pred_test_err = self.predictor(torch.Tensor([standardized_ntk, standardized_lr]).type(torch.float))

#         # if ntk > 1e7:
#         #     valid_err = 100
#         # else:
#         #     data = self.api.query_by_index(index, 'cifar10-valid', hp=200)
#         #     first_trial_seed = list(data.keys())[0]
#         #     valid_acc = data[first_trial_seed].get_eval('valid', 25)['accuracy']
#         #     valid_err = 100 - valid_acc
#         #     if self.current_gen == algorithm.n_gen:
#         #       self.score_dict['n_eval'][self.current_gen] += 1
#         #     else:
#         #       self.score_dict['n_eval'][algorithm.n_gen] = 1
#         #       self.current_gen = algorithm.n_gen
#         # lr = (-1. * lrs.mean()) + (1. * lrs.std())

#         F = [_flops, pred_test_err.item()]

#         self.logger.info(self.arch_info.format(index, *F))
#         out['F'] = np.column_stack(F)



# from pymoo.util.normalization import normalize, standardize
# class TSSGFNorm(TSSBench201):
#     def __init__(self, 
#                 n_repeats=3,
#                 dataset='cifar10',
#                 input_size=(32, 3, 32, 32),
#                 cuda=True,
#                 seed=0, 
#                 **kwargs):

#         super().__init__(n_obj=2,
#                          n_constr=2,
#                          obj_list=['flops', 'ntk', 'lr'],
#                          dataset=dataset,
#                          elementwise_evaluation=False,
#                          **kwargs) 
#         self.ntk_lr_metrics = GradientFreeEvaluator(input_size, dataset, n_repeats, cuda, seed)
#         self.device = torch.device('cuda:0' if cuda else 'cpu')

#         self.min_ntk, self.min_lr = np.inf, np.inf
#         self.max_ntk, self.max_lr = -np.inf, -np.inf

#     def _evaluate(self, X, out, *args, **kwargs):
#         flops, ntks, lrs = [], [], []
#         indices = []
#         for x in X:
#             genotype = x
#             phenotype = self._decode(genotype)
#             index = self.api.query_index_by_arch(phenotype)
#             indices += [index]
#             config = self.api.get_net_config(index, self.dataset)
#             network = get_cell_based_tiny_net(edict(config)).to(self.device)

#             cost_info = self.api.get_cost_info(index, self.dataset)

#             _flops = cost_info['flops']
#             ntk = self.ntk_lr_metrics.calc_ntk(network)
#             lr = self.ntk_lr_metrics.calc_lrc(network)

#             if index in self.score_dict:
#                 self.score_dict[index]['ntks'] += ntk
#                 self.score_dict[index]['lrs'] += lr
#             else:
#                 self.score_dict[index] = {
#                     'ntks': ntk,
#                     'lrs': lr
#                 }
            
#             flops += [_flops]
#             ntks += [np.array(self.score_dict[index]['ntks']).max()]
#             lrs += [np.array(self.score_dict[index]['lrs']).min()]

#             self.max_lr = max(self.max_lr, max(self.score_dict[index]['lrs']))
#             self.min_lr = min(self.min_lr, min(self.score_dict[index]['lrs']))

#             self.max_ntk = max(self.max_ntk, max(self.score_dict[index]['ntks']))
#             self.min_ntk = min(self.min_ntk, min(self.score_dict[index]['ntks']))

#             self.logger.info(
#                 self.arch_info.format(
#                     index, _flops, *list(self.score_dict[index].values())
#                 )
#             )


#         norm_ntks = normalize(ntks, self.min_ntk, self.max_ntk)
#         norm_lrs = normalize(lrs, self.min_lr, self.max_lr)

#         out['F'] = np.column_stack([flops, norm_ntks-norm_lrs])



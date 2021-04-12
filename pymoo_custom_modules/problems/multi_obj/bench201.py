from utils.neural_net.get_ntk import get_ntk_n
from utils.neural_net.kaiming_norm import init_model
from utils.neural_net.linear_region_counter import Linear_Region_Collector

import numpy as np

import torch

from nats_bench import create

import computational_graphs.models.cell_operations as ops

from computational_graphs.models import get_cell_based_tiny_net

import data_loaders

from easydict import EasyDict as edict

import os

from pymoo.model.problem import Problem


import logging

from abc import abstractmethod


class NasBench201(Problem):
    def __init__(self, 
                n_var, 
                n_obj, 
                search_space, 
                iepoch,
                obj_list,
                dataset, 
                elementwise_evaluation=True,
                **kwargs):

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=0,
                         xl=np.zeros(n_var),
                         xu=np.ones(n_var),
                         elementwise_evaluation=elementwise_evaluation,
                         **kwargs)

        self.iepoch = iepoch
        self.hp = None
        self.search_space = search_space
        self.dataset = dataset
        self.obj_list = obj_list
        self.api = create(None, self.search_space, fast_mode=True, verbose=False)
        self.predefined_ops = np.array(ops.NAS_BENCH_201)
        self.dataset = dataset.lower()
        self.model = self._construct_problem()

        self.arch_info = \
            'idx: {} - ' + \
            ' - '.join('{}: '.format(f.replace('accuracy', 'error')) + '{}' for f in self.obj_list)

        self.logger = logging.getLogger(name=self.__class__.__name__)
        self.can_pickle = False
        self.score_dict = {}

    def _evaluate(self, x, out, *args, **kwargs):
        genotype = x
        phenotype = self._decode(genotype)
        index = self.api.query_index_by_arch(phenotype)

        if index in self.score_dict:
            out['F'] = self.score_dict[index]['F']
            self.score_dict[index]['n'] += 1
            self.logger.info('Re-evaluated arch: {} - {} times'.format(index, self.score_dict[index]['n']))
            return
        
        flops = self.api.get_cost_info(index, self.dataset, self.hp)['flops']
        data = self.api.query_by_index(index, self.dataset, hp=self.hp)
        first_trial_seed = list(data.keys())[1]
        valid_acc = data[first_trial_seed].get_eval('valid', self.iepoch)['accuracy']
        valid_err = 100 - valid_acc
        F = [flops, valid_err]

        self.logger.info(self.arch_info.format(index, *F))
        out['F'] = np.column_stack(F)
        self.score_dict[index] = {'F': out['F'], 'n': 1}

    @abstractmethod
    def _decode(self, genotype):
        raise NotImplementedError

    @abstractmethod
    def _construct_problem(self):
        raise NotImplementedError

class SSSBench201(NasBench201):
    def __init__(self, 
                 n_obj=2, 
                 iepoch='12', 
                 obj_list=['flops', 'test-accuracy'], 
                 dataset='cifar10', 
                 **kwargs):
        self.n_bits_per_channel = 3
        
        self.channels_list = np.array([8, 16, 24, 32, 40, 48, 56, 64])
        super().__init__(n_var=15, 
                         n_obj=n_obj, 
                         search_space='sss', 
                         iepoch=iepoch, 
                         obj_list=obj_list, 
                         dataset=dataset, 
                         **kwargs)
        self.hp = 90
        

    def _decode(self, genotype):
        b2i = lambda a: int(''.join(str(int(bit)) for bit in a), 2)
        decoded_indices = [b2i(genotype[start:start+self.n_bits_per_channel]) for start in np.arange(genotype.shape[0])[::self.n_bits_per_channel]]
        channels = self.channels_list[decoded_indices]
        return ':'.join(str(channel) for channel in channels)

    def _construct_problem(self):
        problem = []
        indices = np.arange(self.n_var)
        for i in indices[::self.n_bits_per_channel]:
            problem += [indices[i:i+self.n_bits_per_channel]]
        return problem

class TSSBench201(NasBench201):
    def __init__(self,
                 n_obj=2, 
                 obj_list=['flops', 'valid-accuracy'], 
                 iepoch=11,
                 dataset='cifar10', 
                 **kwargs):
        self.max_nodes = 4
        self.n_bits_per_op = 3
        super().__init__(n_var=18, 
                         n_obj=n_obj, 
                         search_space='tss', 
                         iepoch=iepoch, 
                         obj_list=obj_list, 
                         dataset=dataset, 
                         **kwargs)
        self.hp = 200

    def _decode(self, genotype):
        b2i = lambda a: int(''.join(str(int(bit)) for bit in a), 2)
        decoded_indices = [b2i(genotype[start:start+self.n_bits_per_op]) for start in np.arange(genotype.shape[0])[::self.n_bits_per_op]]
        ops = self.predefined_ops[decoded_indices]
        # ops = self.predefined_ops[genotype]
        node_str = '{}~{}'
        phenotype = []
        k = 0
        for i in range(self.max_nodes):
            node_op = []
            for j in range(i):
                node_op += [node_str.format(ops[k], j)]
                k += 1
            if len(node_op) > 0:
                phenotype += ['|' + '|'.join(node_op) + '|']
                
        phenotype = '+'.join(phenotype)
        return phenotype

    def _construct_problem(self):
        problem = []
        indices = np.arange(self.n_var)
        for i in indices[::self.n_bits_per_op]:
            problem += [indices[i:i+3]]
        return problem

class GradientFreeMetric:
    def __init__(self, 
                 input_size, 
                 dataset, 
                 n_repeats=3, 
                 cuda=True,
                 strategy='avg', 
                 seed=1) -> None:
        self.dataset = dataset
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.n_repeats = n_repeats

        self.lrc_model = Linear_Region_Collector(
                          input_size=(1000, 3, 3, 3), 
                          sample_batch=3, 
                          dataset=dataset,
                          data_path='~/.torch/',
                          seed=seed)
        self.loader = getattr(data_loaders, dataset.upper())(
            data_folder='~/.torch/',
            num_workers=4,
            batch_size=input_size[0],
            pin_memory=True,
            drop_last=True
        ).train_loader

        self.ntk_strategy = np.mean if strategy == 'avg' else max
        self.lr_strategy = np.mean if strategy == 'avg' else min

    def calc_ntk(self, network):
        NTK = []
        for _ in range(self.n_repeats):
            network = init_model(network, method='kaiming_norm_fanout')
            ntk = get_ntk_n(self.loader, [network], recalbn=0, train_mode=True, num_batch=1)
            NTK += ntk
        network.zero_grad()
        torch.cuda.empty_cache()
        return NTK

    def calc_lrc(self, network):
        LR = []
        network.train()
        with torch.no_grad():
            for _ in range(self.n_repeats):
                network = init_model(network, method='kaiming_norm_fanin')
                self.lrc_model.reinit([network])
                lr = self.lrc_model.forward_batch_sample()
                LR += lr
                self.lrc_model.clear()

        torch.cuda.empty_cache()
        return LR


from pymoo.util.normalization import normalize
class TSSGFNorm(TSSBench201):
    def __init__(self, 
                n_repeats=3,
                dataset='cifar10',
                input_size=(32, 3, 32, 32),
                cuda=True,
                seed=0, 
                **kwargs):

        super().__init__(n_obj=2,
                         n_constr=2,
                         obj_list=['flops', 'ntk', 'lr'],
                         dataset=dataset,
                         elementwise_evaluation=False,
                         **kwargs) 
        self.ntk_lr_metrics = GradientFreeMetric(input_size, dataset, n_repeats, cuda, seed)
        self.device = torch.device('cuda:0' if cuda else 'cpu')

        self.min_ntk, self.min_lr = np.inf, np.inf
        self.max_ntk, self.max_lr = -np.inf, -np.inf

    def _evaluate(self, X, out, *args, **kwargs):
        flops, ntks, lrs = [], [], []
        indices = []
        for x in X:
            genotype = x
            phenotype = self._decode(genotype)
            index = self.api.query_index_by_arch(phenotype)
            indices += [index]
            config = self.api.get_net_config(index, self.dataset)
            network = get_cell_based_tiny_net(edict(config)).to(self.device)

            cost_info = self.api.get_cost_info(index, self.dataset)

            _flops = cost_info['flops']
            ntk = self.ntk_lr_metrics.calc_ntk(network)
            lr = self.ntk_lr_metrics.calc_lrc(network)

            if index in self.score_dict:
                self.score_dict[index]['ntks'] += ntk
                self.score_dict[index]['lrs'] += lr
            else:
                self.score_dict[index] = {
                    'ntks': ntk,
                    'lrs': lr
                }
            
            flops += [_flops]
            ntks += [np.array(self.score_dict[index]['ntks']).max()]
            lrs += [np.array(self.score_dict[index]['lrs']).min()]

            self.max_lr = max(self.max_lr, max(self.score_dict[index]['lrs']))
            self.min_lr = min(self.min_lr, min(self.score_dict[index]['lrs']))

            self.max_ntk = max(self.max_ntk, max(self.score_dict[index]['ntks']))
            self.min_ntk = min(self.min_ntk, min(self.score_dict[index]['ntks']))

            self.logger.info(
                self.arch_info.format(
                    index, _flops, *list(self.score_dict[index].values())
                )
            )


        norm_ntks = normalize(ntks, self.min_ntk, self.max_ntk)
        norm_lrs = normalize(lrs, self.min_lr, self.max_lr)

        out['F'] = np.column_stack([flops, norm_ntks-norm_lrs])


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
                hp,
                obj_list,
                dataset, 
                **kwargs):

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=0,
                         xl=np.zeros(n_var),
                         xu=np.ones(n_var),
                         elementwise_evaluation=True,
                         **kwargs)

        self.hp = hp
        self.search_space = search_space
        self.dataset = dataset
        self.obj_list = obj_list
        self.api = create('NATS-tss-v1_0-3ffb9-simple', self.search_space, fast_mode=True, verbose=False)
        self.predefined_ops = np.array(ops.NAS_BENCH_201)
        self.dataset = dataset.lower()
        self.model = self._construct_problem()

        self.arch_info = \
            'idx: {} - ' + \
            ' - '.join('{}: '.format(f.replace('accuracy', 'error')) + '{}' for f in self.obj_list)

        self.logger = logging.getLogger(name=self.__class__.__name__)
        self.can_pickle = False

    def _get_pickle_obj(self):
        api = self.api
        self.api = None
        return self, api
    
    def _load(self, api):
        self.api = api
        return self

    def _evaluate(self, x, out, *args, **kwargs):
        genotype = x
        phenotype = self._decode(genotype)
        index = self.api.query_index_by_arch(phenotype)

        cost_info = self.api.get_cost_info(index, self.dataset)
        more_info = self.api.get_more_info(index, self.dataset, is_random=True)
        F = []
        for f in self.obj_list:
            if f in cost_info.keys():
                F += [cost_info[f]]
            else:
                F += [100 - more_info[f]] if 'accuracy' in f else [more_info[f]]
        self.logger.info(self.arch_info.format(index, *F))
        out['F'] = np.column_stack(F)

    @abstractmethod
    def _decode(self, genotype):
        raise NotImplementedError

    @abstractmethod
    def _construct_problem(self):
        raise NotImplementedError

class SSSBench201(NasBench201):
    def __init__(self, 
                 n_obj=2, 
                 hp='12', 
                 obj_list=['flops', 'train-accuracy'], 
                 dataset='cifar10', 
                 **kwargs):
        self.n_bits_per_channel = 3
        self.channels_list = np.array([8, 16, 24, 32, 40, 48, 56, 64])
        super().__init__(n_var=15, 
                         n_obj=n_obj, 
                         search_space='sss', 
                         hp=hp, 
                         obj_list=obj_list, 
                         dataset=dataset, 
                         **kwargs)
        

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
                 obj_list=['flops', 'train-accuracy'], 
                 hp='12',
                 dataset='cifar10', 
                 **kwargs):
        self.max_nodes = 4
        self.n_bits_per_op = 3
        super().__init__(n_var=18, 
                         n_obj=n_obj, 
                         search_space='tss', 
                         hp=hp, 
                         obj_list=obj_list, 
                         dataset=dataset, 
                         **kwargs)
        

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



class TSSBench201GradientFree(TSSBench201):
    def __init__(self, 
                n_repeats=3,
                dataset='cifar10',
                hp='12',
                input_size=(32, 3, 32, 32),
                cuda=True,
                seed=0, 
                **kwargs):

        super().__init__(n_obj=3,
                         obj_list=['flops', 'ntk', 'lr'],
                         hp=hp,
                         dataset=dataset,
                         **kwargs) 
        self.n_repeats = n_repeats
        self.input_size = input_size
        self.device = torch.device('cuda:0' if cuda else 'cpu')

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

    def _evaluate(self, x, out, *args, **kwargs):
        genotype = x
        phenotype = self._decode(genotype)
        index = self.api.query_index_by_arch(phenotype)
        config = self.api.get_net_config(index, self.dataset)
        network = get_cell_based_tiny_net(edict(config)).to(self.device)

        cost_info = self.api.get_cost_info(index, self.dataset, hp=self.hp)

        F = [cost_info['flops'], self.__calc_ntk(network), self.__calc_lrc(network)]

        self.logger.info(self.arch_info.format(index, *F))
        out['F'] = np.column_stack(F)


    def __calc_ntk(self, network, **kwargs):
        NTK = []
        for _ in range(self.n_repeats):
            network = init_model(network, method='kaiming_norm_fanout')
            ntk = get_ntk_n(self.loader, [network], recalbn=0, train_mode=True, num_batch=1)
            NTK += ntk
        network.zero_grad()
        torch.cuda.empty_cache()
        return np.max(NTK)

    def __calc_lrc(self, network, **kwargs):
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
        return -np.min(LR)

    def _get_pickle_obj(self):
        api = self.api
        self.api = None
        loader = self.loader
        self.loader = None
        lrc_model = self.lrc_model
        self.lrc_model = None
        
        return self, [api, loader, lrc_model]

    def _load(self, obj):
        self.api, self.loader, self.lrc_model = obj
        return self

class SSSBench201GradientFree(SSSBench201):
    def __init__(self, 
                n_repeats=3,
                dataset='cifar10',
                hp='12',
                input_size=(32, 3, 32, 32),
                cuda=True,
                seed=0, 
                **kwargs):

        super().__init__(n_obj=3,
                         obj_list=['flops', 'ntk', 'lr'],
                         hp=hp,
                         dataset=dataset,
                         **kwargs) 
        self.n_repeats = n_repeats
        self.input_size = input_size
        self.device = torch.device('cuda:0' if cuda else 'cpu')

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

    def _evaluate(self, x, out, *args, **kwargs):
        genotype = x
        phenotype = self._decode(genotype)
        index = self.api.query_index_by_arch(phenotype)
        config = self.api.get_net_config(index, self.dataset)
        network = get_cell_based_tiny_net(edict(config)).to(self.device)

        cost_info = self.api.get_cost_info(index, self.dataset, hp=self.hp)

        F = [cost_info['flops'], self.__calc_ntk(network), self.__calc_lrc(network)]

        self.logger.info(self.arch_info.format(index, *F))
        out['F'] = np.column_stack(F)


    def __calc_ntk(self, network, **kwargs):
        NTK = []
        for _ in range(self.n_repeats):
            network = init_model(network, method='kaiming_norm_fanout')
            ntk = get_ntk_n(self.loader, [network], recalbn=0, train_mode=True, num_batch=1)
            NTK += ntk
        network.zero_grad()
        torch.cuda.empty_cache()
        return np.max(NTK)

    def __calc_lrc(self, network, **kwargs):
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
        return -np.min(LR)
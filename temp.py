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
                bench_path,
                obj_list,
                dataset,
                n_constr=0,
                elementwise_evaluation=True, 
                **kwargs):

        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         n_constr=n_constr,
                         xl=np.zeros(n_var),
                         xu=np.ones(n_var),
                         elementwise_evaluation=elementwise_evaluation,
                         **kwargs)

        self.hp = hp
        self.search_space = search_space
        self.dataset = dataset
        self.obj_list = obj_list
        self.api = create(os.getcwd() + bench_path, self.search_space, fast_mode=True, verbose=False)
        self.predefined_ops = np.array(ops.NAS_BENCH_201)
        self.dataset = dataset.lower()
        self.model = self._construct_problem()

        self.arch_info = \
            'idx: {} - ' + \
            ' - '.join('{}: '.format(f.replace('accuracy', 'error')) + '{}' for f in self.obj_list)

        self.logger = logging.getLogger(name=self.__class__.__name__)
        self.score_dict = {}

    def _evaluate(self, x, out, *args, **kwargs):
        genotype = x
        phenotype = self._decode(genotype)
        index = self.api.query_index_by_arch(phenotype)
        
        cost_info = self.api.get_cost_info(index, self.dataset, hp=self.hp)
        more_info = self.api.get_more_info(index, self.dataset, hp=self.hp, is_random=False)
        F = []
        for f in self.obj_list:
            if f in cost_info.keys():
                F += [cost_info[f]]
            else:
                F += [100 - more_info[f]] if 'accuracy' in f else [more_info[f]]
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
                 bench_path='/NATS-sss-v1_0-50262-simple/',
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
                         bench_path=bench_path,
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

class TSSBench201(NasBench201):
    def __init__(self,
                 bench_path='/NATS-tss-v1_0-3ffb9-simple/',
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
                         bench_path=bench_path,
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

class GradientFreeMetric:
    def __init__(self, 
                 input_size, 
                 dataset, 
                 n_repeats=3, 
                 strategy='avg', 
                 seed=1,
                 num_workers=2) -> None:
        self.dataset = dataset
        self.n_repeats = n_repeats

        self.lrc_model = Linear_Region_Collector(
                          input_size=(1000, 1, 3, 3), 
                          sample_batch=3, 
                          dataset=dataset,
                          data_path=os.getcwd(),
                          seed=seed,
                          num_workers=num_workers)
        self.loader = getattr(data_loaders, dataset.upper())(
            data_folder=os.getcwd(),
            num_workers=num_workers,
            batch_size=input_size[0],
            pin_memory=True,
            drop_last=True,
            worker_init_fn=random.seed(seed)
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
        # return self.ntk_strategy(NTK)
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
        # return -self.lr_strategy(LR)
        return LR

import random
class TSSBench201GradientFree(TSSBench201):
    def __init__(self, 
                n_repeats=3,
                dataset='cifar10',
                hp='12',
                input_size=(32, 3, 32, 32),
                cuda=True,
                seed=0, 
                num_workers=4,
                **kwargs):

        super().__init__(n_obj=3,
                         obj_list=['flops', 'ntk', 'lr'],
                         hp=hp,
                         dataset=dataset,
                         elementwise_evaluation=True,
                         **kwargs) 
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.ntk_lr_metrics = GradientFreeMetric(input_size, dataset, n_repeats, 'best', seed, num_workers)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True

    def _evaluate(self, x, out, *args, **kwargs):
        genotype = x
        phenotype = self._decode(genotype)
        index = self.api.query_index_by_arch(phenotype)
        # if index in self.score_dict:
        #   if self.score_dict[index]['n'] == 0:
        #     pass
        #   out['F'] = self.score_dict[index]['F']
        #   self.score_dict[index]['n'] += 1
        #   self.logger.info('Re-evaluate arch: {} {} times'.format(index, self.score_dict[index]['n']))
        #   return
        
        config = self.api.get_net_config(index, self.dataset)
        config['C'] = 3
        config['N'] = 1
        config['depth'] = -1
        config_thin = config.copy()
        config_thin['C'] = config_thin['C_in'] = config_thin['depth'] = config_thin['N'] = 1
        network = get_cell_based_tiny_net(edict(config)).to(self.device)
        network_thin = get_cell_based_tiny_net(edict(config_thin)).to(self.device)

        cost_info = self.api.get_cost_info(index, self.dataset)

        _flops = cost_info['flops']
        ntk = np.array(self.ntk_lr_metrics.calc_ntk(network)).std()
        lr = -np.array(self.ntk_lr_metrics.calc_lrc(network_thin)).min()

        F = [cost_info['flops'], ntk, lr]

        self.logger.info(self.arch_info.format(index, *F))
        out['F'] = np.column_stack(F)
        self.score_dict[index] = {'F': out['F'], 'n': 1}

class TSSGFRank(TSSBench201):
    def __init__(self, 
                n_repeats=3,
                dataset='cifar10',
                input_size=(32, 3, 32, 32),
                cuda=True,
                seed=0, 
                num_workers=2,
                **kwargs):

        super().__init__(n_obj=2,
                         n_constr=0,
                         obj_list=['flops', 'ntk', 'lr'],
                         dataset=dataset,
                         elementwise_evaluation=False,
                         **kwargs) 
        self.ntk_lr_metrics = GradientFreeMetric(input_size, dataset, n_repeats, cuda, seed, num_workers)
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.global_ntks = []
        self.global_lrs = []
        self.global_indices = []

    def _evaluate(self, X, out, *args, **kwargs):
        flops, ntks, lrs = [], [], []
        indices = []
        for x in X:
            genotype = x
            phenotype = self._decode(genotype)
            index = self.api.query_index_by_arch(phenotype)
            indices += [index]
            config = self.api.get_net_config(index, self.dataset)
            config['C'] = 3
            config['N'] = 1
            config['depth'] = -1
            config_thin = config.copy()
            config_thin['C'] = config_thin['C_in'] = config_thin['depth'] = config_thin['N'] = 1
            network = get_cell_based_tiny_net(edict(config)).to(self.device)
            network_thin = get_cell_based_tiny_net(edict(config_thin)).to(self.device)

            cost_info = self.api.get_cost_info(index, self.dataset)

            _flops = cost_info['flops']
            ntk = self.ntk_lr_metrics.calc_ntk(network)
            lr = self.ntk_lr_metrics.calc_lrc(network_thin)

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

            self.logger.info(self.arch_info.format(index, _flops, *list(self.score_dict[index].values())))

        ntks = np.array(ntks)
        lrs = np.array(lrs)
        indices = np.array(indices)

        # #### Global Rank #####
        # self.global_ntks = np.concatenate([self.global_ntks, ntks])
        # self.global_lrs = np.concatenate([self.global_lrs, lrs])
        # self.global_indices = np.concatenate([self.global_indices, indices])
        # _, unique_indices = np.unique(self.global_indices, return_index=True)
        # self.global_ntks = self.global_ntks[unique_indices]
        # self.global_lrs = self.global_lrs[unique_indices]
        # self.global_indices = self.global_indices[unique_indices]
        

        # global_ntk_ranks = self.global_ntks.argsort().argsort()
        # global_lr_ranks = self.global_lrs.argsort()[::-1].argsort()

        # sorted_indices = self.global_indices.argsort()
        # indices = sorted_indices[np.searchsorted(self.global_indices, indices, sorter=sorted_indices)]
        # ntk_ranks = global_ntk_ranks[indices]
        # lr_ranks = global_lr_ranks[indices]
        # ranks = ntk_ranks + lr_ranks
        # ##### Global Rank #####

        ##### Local Rank #####
        # _, unique = np.unique(indices, return_index=True)
        # unique_ntks = ntks[unique]
        # unique_lrs = lrs[unique]
        # unique_indices = indices[unique]
        # ntk_ranks = unique_ntks.argsort().argsort()
        # lr_ranks = unique_lrs.argsort()[::-1].argsort()
        # sorted_unique_indices = unique_indices.argsort()
        # indices = sorted_unique_indices[np.searchsorted(unique_indices, indices, sorter=sorted_unique_indices)]
        # ntk_ranks = ntk_ranks[indices]
        # lr_ranks = lr_ranks[indices]
        # ranks = ntk_ranks + lr_ranks
        ##### Local Rank #####

        # self.global_ntks[indices] = ntks
        # self.global_lrs[indices] = lrs

        # global_ntk_ranks = self.global_ntks.argsort().argsort()
        # global_lr_ranks = self.global_lrs.argsort()[::-1].argsort()

        # ntk_ranks = global_ntk_ranks[indices]
        # lr_ranks = global_lr_ranks[indices]
        ntk_ranks = ntks.argsort().argsort()
        lr_ranks = lrs.argsort()[::-1].argsort()
        self.logger.info('ntk ranks: {}'.format(ntk_ranks))
        self.logger.info('lr ranks: {}'.format(lr_ranks))
        ranks = ntk_ranks + lr_ranks

        out['F'] = np.column_stack([flops, ranks])

from pymoo.util.normalization import normalize
class TSSGFNorm(TSSBench201):
    def __init__(self, 
                n_repeats=3,
                dataset='cifar10',
                input_size=(32, 3, 32, 32),
                cuda=True,
                num_workers=2,
                seed=0, 
                **kwargs):

        super().__init__(n_obj=2,
                         n_constr=0,
                         obj_list=['flops', 'ntk', 'lr'],
                         dataset=dataset,
                         elementwise_evaluation=False,
                         **kwargs) 
        self.ntk_lr_metrics = GradientFreeMetric(input_size, dataset, n_repeats, cuda, seed, num_workers)
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        # self.global_ntks = np.zeros(5**6)
        # self.global_lrs = np.zeros(5**6)
        self.max_lr = self.max_ntk = -np.inf
        self.min_lr = self.min_ntk = np.inf

    def _evaluate(self, X, out, *args, **kwargs):
        flops, ntks, lrs = [], [], []
        indices = []
        for x in X:
            genotype = x
            phenotype = self._decode(genotype)
            index = self.api.query_index_by_arch(phenotype)
            indices += [index]
            config = self.api.get_net_config(index, self.dataset)
            config['C'] = 3
            config['N'] = 1
            config['depth'] = -1
            config_thin = config.copy()
            config_thin['C'] = config_thin['C_in'] = config_thin['depth'] = config_thin['N'] = 1
            network = get_cell_based_tiny_net(edict(config)).to(self.device)
            network_thin = get_cell_based_tiny_net(edict(config_thin)).to(self.device)

            cost_info = self.api.get_cost_info(index, self.dataset)

            _flops = cost_info['flops']
            ntk = self.ntk_lr_metrics.calc_ntk(network)
            lr = self.ntk_lr_metrics.calc_lrc(network_thin)

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


            self.logger.info(self.arch_info.format(index, _flops, *list(self.score_dict[index].values())))
            
        ntks = np.array(ntks)
        lrs = np.array(lrs)
        indices = np.array(indices)

        self.max_ntk = max(self.max_ntk, max(ntks))
        self.max_lr = max(self.max_lr, max(lrs))
        self.min_lr = min(self.min_lr, min(lrs))
        self.min_ntk = min(self.min_ntk, min(ntks))

        # self.global_ntks[indices] = ntks
        # self.global_lrs[indices] = lrs

        self.logger.info('lr max: {} min {}'.format(self.max_lr, self.min_lr))
        self.logger.info('ntk max: {} min {}'.format(self.max_ntk, self.min_ntk))

        norm_ntks = normalize(ntks, self.min_ntk, self.max_ntk)
        norm_lrs = normalize(lrs, self.min_lr, self.max_lr)

        out['F'] = np.column_stack([flops, norm_ntks-norm_lrs])

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
                          data_path=os.getcwd(),
                          seed=seed)
        self.loader = getattr(data_loaders, dataset.upper())(
            data_folder=os.getcwd(),
            num_workers=4,
            batch_size=input_size[0],
            pin_memory=True,
            drop_last=True
        ).train_loader

        self.score_dict = {}

    def _evaluate(self, x, out, *args, **kwargs):
        genotype = x
        phenotype = self._decode(genotype)
        index = self.api.query_index_by_arch(phenotype)
        if index in self.score_dict:
          out['F'] = self.score_dict[index]['F']
          self.score_dict[index]['n'] += 1
          self.logger.info('Re-evaluate arch: {} {} times'.format(index, self.score_dict[index]['n']))
          return
        
        config = self.api.get_net_config(index, self.dataset)
        network = get_cell_based_tiny_net(edict(config)).to(self.device)

        cost_info = self.api.get_cost_info(index, self.dataset, hp=self.hp)

        F = [cost_info['flops'], self.__calc_ntk(network), self.__calc_lrc(network)]

        self.logger.info(self.arch_info.format(index, *F))
        out['F'] = np.column_stack(F)
        self.score_dict[index] = {'F': out['F'], 'n': 1}


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
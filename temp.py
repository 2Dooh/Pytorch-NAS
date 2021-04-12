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
                          input_size=(1000, 3, 3, 3), 
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
                         elementwise_evaluation=True,
                         **kwargs) 
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.ntk_lr_metrics = GradientFreeMetric(input_size, dataset, n_repeats, 'best', seed)

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
        network = get_cell_based_tiny_net(edict(config)).to(self.device)

        cost_info = self.api.get_cost_info(index, self.dataset, hp=self.hp)

        F = [cost_info['flops'], self.ntk_lr_metrics.calc_ntk(network), self.ntk_lr_metrics.calc_lrc(network)]

        self.logger.info(self.arch_info.format(index, *F))
        out['F'] = np.column_stack(F)
        # self.score_dict[index] = {'F': out['F'], 'n': 1}

        
        # flops, ntks, lrs = [], [], []
        # for x in X:
        #     genotype = x
        #     phenotype = self._decode(genotype)
        #     index = self.api.query_index_by_arch(phenotype)

        #     if index in self.score_dict:
        #         _flops, ntk, lr = self.score_dict[index]['F']
        #         flops += [_flops]
        #         ntks += [ntk]
        #         lrs += [lr]
        #         self.score_dict[index]['n'] += 1
        #         self.logger.info('Re-evaluate arch : {} - {} times'.format(index, 
        #                                                                    self.score_dict[index]['n']))
        #         continue

        #     config = self.api.get_net_config(index, self.dataset)
        #     network = get_cell_based_tiny_net(edict(config)).to(self.device)

        #     _flops = self.api.get_cost_info(index, self.dataset, hp=self.hp)['flops']
        #     ntk, lr = self.ntk_lr_metrics.calc_ntk(network), self.ntk_lr_metrics.calc_lrc(network)

        #     flops.append(_flops)
        #     ntks.append(ntk)
        #     lrs.append(lr)

        #     self.score_dict[index] = {'F': [_flops, ntk, lr], 'n': 1}

        #     self.logger.info(self.arch_info.format(index, *self.score_dict[index]['F']))

        # # self.logger.info(flops)
        # flops = np.array(flops)
        # rank_ntks = np.argsort(ntks)
        # rank_lrs = np.argsort(lrs)[::-1]
        # relative_rank = (rank_ntks + rank_lrs).astype(flops.dtype)

        # out['F'] = np.column_stack([flops, relative_rank])

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
        self.global_ntks = np.ones(5**6) * np.inf
        self.global_lrs = np.ones(5**6) * np.inf

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
            

            flops += [_flops]
            if index in self.score_dict:
                ntks += [max(self.score_dict[index]['ntks'])]
                lrs += [min(self.score_dict[index]['lrs'])]
                self.score_dict[index]['n'] += 1
            else:
                ntk = self.ntk_lr_metrics.calc_ntk(network)
                lr = self.ntk_lr_metrics.calc_lrc(network)
                self.score_dict[index] = {
                    'ntks': ntk,
                    'lrs': lr,
                    'n': 1
                }
                ntks += [max(self.score_dict[index]['ntks'])]
                lrs += [min(self.score_dict[index]['lrs'])]

            self.logger.info(self.arch_info.format(index, _flops, *list(self.score_dict[index].values())))

        ntks = np.array(ntks)
        lrs = np.array(lrs)
        indices = np.array(indices)

        self.logger.info('lr max: {} min {} mean {} std {}'.format(lrs.max(), lrs.min(), lrs.mean(), lrs.std()))
        self.logger.info('ntk max: {} min {} mean {} std {}'.format(ntks.max(), ntks.min(), ntks.mean(), ntks.std()))

        self.global_ntks[indices] = ntks
        self.global_lrs[indices] = lrs

        global_ntk_ranks = np.argsort(self.global_ntks)
        global_lr_ranks = np.argsort(self.global_lrs)[::-1]

        ntk_ranks = global_ntk_ranks[indices]
        lr_ranks = global_lr_ranks[indices]
        ranks = ntk_ranks + lr_ranks

        out['F'] = np.column_stack([flops, ranks])
        self.logger.info(out['F'])

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
        self.global_ntks = np.zeros(5**6)
        self.global_lrs = np.zeros(5**6)

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
            ntks += [max(self.score_dict[index]['ntks'])]
            lrs += [min(self.score_dict[index]['lrs'])]

            self.logger.info(self.arch_info.format(index, _flops, *list(self.score_dict[index].values())))
        ntks = np.array(ntks)
        lrs = np.array(lrs)
        indices = np.array(indices)

        self.global_ntks[indices] = ntks
        self.global_lrs[indices] = lrs

        self.logger.info('lr max: {} min {}'.format(self.global_lrs.max(), self.global_lrs.min()))
        self.logger.info('ntk max: {} min {}'.format(self.global_ntks.max(), self.global_ntks.min()))

        norm_ntks = normalize(ntks, 0, self.global_ntks.max())
        norm_lrs = normalize(lrs, 0, self.global_lrs.max())

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
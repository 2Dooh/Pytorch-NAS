import computational_graphs.estimators.problems.multi_obj.mo_problem as base

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

class NasGradientFree(base.MultiObjectiveProblem):
    def __init__(self, n_repeats=3, **kwargs):
        super().__init__(**kwargs)
        self.n_repeats = n_repeats
        

class NasBench201GradientFree(NasGradientFree):
    def __init__(self, 
                dataset, 
                max_nodes, 
                input_size=(32, 3, 32, 32),
                cuda=True, 
                **kwargs):
        super().__init__(n_params=(max_nodes * max_nodes-1) // 2,
                        n_obj=2,
                        constraints=0,
                        type=np.int,
                        **kwargs)

        self.max_nodes = max_nodes
        self.api = create(None, 'tss', fast_mode=True, verbose=False)
        self.predefined_ops = np.array(ops.NAS_BENCH_201)
        xl = np.zeros(self.n_params)
        xu = np.ones(self.n_params) * (len(self.predefined_ops)-1)
        self.domain = (xl, xu)

        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.dataset = dataset.lower()

        self.loader = getattr(data_loaders, dataset.upper())(
            data_folder='~/.torch/',
            num_workers=4,
            batch_size=input_size[0],
            pin_memory=True,
            drop_last=True
        ).train_loader
        self.input_size = input_size
        self.global_ntk = []
        self.global_lrc = []

    def _f(self, X):
        f_X = []
        for i in range(X.shape[0]):
            genotype = X[i]
            phenotype = self.__decode(genotype)
            index = self.api.query_index_by_arch(phenotype)
            f2_x = self.api.get_cost_info(index, self.dataset)['flops']
            config = self.api.get_net_config(index, self.dataset)
            network = get_cell_based_tiny_net(edict(config))

            init_model(network)
            network.to(self.device)
            NTK, LRC = [], []
            for _ in range(self.n_repeats):
                ntk = get_ntk_n(
                    self.loader, 
                    [network], 
                    num_batch=1, 
                    train_mode=False
                )
                NTK += ntk

                lrc_model = \
                    Linear_Region_Collector(
                        input_size=self.input_size, 
                        sample_batch=10, 
                        data_loader=self.loader, 
                        seed=1, 
                        models=[network]
                    )
                lrc = lrc_model.forward_batch_sample()
                LRC += lrc
                lrc_model.clear()
            ntk, lrc = np.mean(NTK), np.mean(LRC)
            self.global_ntk += [ntk]
            self.global_lrc += [lrc]
            ntk_rank = sorted(self.global_ntk)
            lrc_rank = sorted(self.global_lrc, reverse=True)
            f1_x = ntk_rank.index(ntk) + lrc_rank.index(lrc)
            f_X += [[f1_x, f2_x]]

        return np.array(f_X).reshape((X.shape[0], -1))

    @staticmethod
    def _compare(y1, y2):
        not_dominated = y1 <= y2
        dominate = y1 < y2
        return not_dominated.all() and True in dominate

    @staticmethod
    def _compare_vectorized(Y1, Y2):
        not_dominated = Y1 <= Y2
        dominate = Y1 < Y2

        pareto_dominants = np.logical_and(
            not_dominated.all(axis=1),
            dominate.any(axis=1)
        ) 
        return pareto_dominants

    def __decode(self, genotype):
        ops = self.predefined_ops[genotype]
        node_str = '|{}~{}'
        phenotype = ''
        k = 0
        for i in range(self.max_nodes):
            for j in range(i):
                phenotype += node_str.format(ops[k], j)
                k += 1
            if i != 0:
                phenotype += '|+'
        
        return phenotype[:-1]
from pickle import load
from .get_ntk import get_ntk_n

from .kaiming_norm import init_model

import torch

from .linear_region_counter import Linear_Region_Collector

import loaders

import random

import os

class GradientFreeEvaluator:
    def __init__(self, 
                 dataset='cifar10', 
                 lr_sample_batch=3,
                 lr_input_size=(1000, 1, 3, 3), 
                 ntk_batch_size=16,
                 seed=1,
                 num_workers=2,
                 n_repeats=3,
                 root_folder='~/.torch') -> None:
        self.dataset = dataset
        self.n_repeats = n_repeats
        self.lrc_model = Linear_Region_Collector(
                          input_size=lr_input_size, 
                          sample_batch=lr_sample_batch, 
                          dataset=dataset,
                          data_path=root_folder,
                          seed=seed,
                          num_workers=num_workers)
        self.loader = getattr(loaders, dataset)(
            data_folder=root_folder,
            num_workers=num_workers,
            batch_size=ntk_batch_size,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=random.seed(seed)
        ).train_loader


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
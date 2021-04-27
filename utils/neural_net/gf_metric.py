from .get_ntk import get_ntk_n

from .kaiming_norm import init_model

import torch

from .linear_region_counter import Linear_Region_Collector

import data_loaders

import random

import os

class GradientFreeMetric:
    def __init__(self, 
                 input_size, 
                 dataset,  
                 seed=1,
                 num_workers=2) -> None:
        self.dataset = dataset

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
            batch_size=16,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=random.seed(seed)
        ).train_loader


    def calc_ntk(self, network, n_repeats):
        NTK = []
        for _ in range(n_repeats):
            network = init_model(network, method='kaiming_norm_fanout')
            ntk = get_ntk_n(self.loader, [network], recalbn=0, train_mode=True, num_batch=1)
            NTK += ntk
        network.zero_grad()
        torch.cuda.empty_cache()
        # return self.ntk_strategy(NTK)
        return NTK

    def calc_lrc(self, network, n_repeats):
        LR = []
        network.train()
        with torch.no_grad():
            for _ in range(n_repeats):
                network = init_model(network, method='kaiming_norm_fanin')
                self.lrc_model.reinit([network])
                lr = self.lrc_model.forward_batch_sample()
                LR += lr
                self.lrc_model.clear()

        torch.cuda.empty_cache()
        # return -self.lr_strategy(LR)
        return LR
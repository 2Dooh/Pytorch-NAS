import torch
from torch.nn import Module, Linear, Sequential
from torch.nn.utils import prune
from torch import nn
import numpy as np
from itertools import product
from .custom_modules.computational_block import *

class EvoNetwork(Module):
    def __init__(self, 
                 config, 
                 encoding, 
                 input_size, 
                 output_size):
        super(EvoNetwork, self).__init__()

        self.model = Decoder(config, encoding).get_model()

        out = self.model(torch.autograd.Variable(torch.zeros(1, *input_size)))
        shape = out.data.shape

        self.gap = ...

        shape = self.gap(out).data.shape

        self.linear = nn.Linear(shape[1]*shape[2]*shape[3], output_size)

        self.model.zero_grad()

    def forward(self, x):
        x = self.gap(self.model(x))
        x = x.flatten()
        return self.linear(x)

class Decoder(object):
    def __call__(self, sample):
        pass

# class SuperNet(Module):
#     def __init__(self, config):
#         super(SuperNet, self).__init__()
#         self.config = config
#         kernel_sizes = [3, 5, 7]    # 2 bit
#         pool_sizes = [1, 2] # 1 bit
#         n_channels = [self.config['input_size'][2], 8, 16, 32, 64, 128]    # 3 bit
#         node_types = [('res', ResidualNode),
#                       ('res_pre', PreactResidualNode),
#                       ('dense', DenseNode)] # 2 bit
#         stages = list(range(config['n_stages']))
#         nodes = list(range(max(config['n_nodes'])))
#         channel_flow = list(combinations_with_replacement(n_channels, 2))
#         choices = [kernel_sizes, channel_flow, node_types]
#         self.module_dict = nn.ModuleDict()
#         for choice in product(*choices):
#             name, module = self.construct_sub_modules(choice)
#             print(name)
#             self.module_dict[name] = module

#         print(len(self.module_dict))


#     def forward(self, x, encoder):
#         temp = []
#         nodes_per_stage = self.config['nodes']
#         for phase, n_nodes in enumerate(nodes_per_stage):
#             start_idx = 0 if phase == 0 else self.__calc_phase_length(nodes_per_stage[phase-1])
#             end_idx = start_idx + self.__calc_phase_length(n_nodes)
#             phase_encoding = encoder[start_idx : end_idx]
            
#             temp += []
#         pass

#     def set_path(self, path):
#         pass

#     def construct_sub_modules(self, config):
#         kernel_size = config[0]
#         (node_type, node) = config[2]
#         (in_channels, out_channels) = config[1]
#         sub_module = node(in_channels, out_channels, kernel_size)
#         name = '{}{}x{}-{}-{}'.format(node_type, 
#                                        kernel_size,
#                                        kernel_size, 
#                                        in_channels, 
#                                        out_channels)
#         return name, sub_module

#     def __calc_phase_length(self, n_nodes):
#         connection_length = (n_nodes * (n_nodes-1)) // 2 + 1
#         block_info_length = 8
#         return block_info_length + connection_length

# class Decoder:
#     pass

# class DemoConfig:
#     def __init__(self):
#         self.n_stages = 3
#         self.n_nodes = 6
#         self.input_size = (32, 32, 3)

# def demo():
#     config = {'n_stages': 3, 'n_nodes' : (6, 6, 6), 'input_size': (32, 32, 3)}
#     supernet = SuperNet(config)
#     pass

# demo()
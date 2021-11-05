import torch
from lib.benchmarks.nasbench101 import nasbench

from lib.benchmarks.nasbench101.nasbench import api
from lib.nasbench_pytorch.model import Network

from utils.neural_net.flops_benchmark import get_model_infos

from os.path import expanduser

import os

import numpy as np

# home_dir = expanduser('~')
# nasbench = api.NASBench(os.path.join(home_dir, '.torch/nasbench_only108.tfrecord'))

# bench_hash = torch.load('data/bench101_hash.pth.tar')

# for key in bench_hash.keys():
#     fixed_stat, computed_stat = nasbench.get_metrics_from_hash(key)
#     bench_hash[key] = {'fixed_stat': fixed_stat, 'computed_stat': computed_stat}
#     matrix = fixed_stat['module_adjacency']
#     ops = fixed_stat['module_operations']

#     spec = api.ModelSpec(
#             matrix=matrix,
#             ops=ops
#         )
#     network = Network(spec, num_labels=10).cuda()
#     flops, _ = get_model_infos(network, shape=[1, 3, 32, 32])

#     bench_hash[key]['flops'] = flops    


# torch.save(bench_hash, 'data/bench101_info.pth.tar', pickle_protocol=5)


# bench_info = torch.load('data/bench101_info.pth.tar')

# bench_front = []
# for key, val in bench_info.items():
#     flops = val['flops']
#     performance_results = val['computed_stat'][108]
#     test_acc = np.mean([performance_results[i]['final_test_accuracy'] for i in range(3)])
#     test_err = (1 - test_acc) * 100.

#     bench_front += [[flops, test_err]]
#     print([flops, test_err])

# bench_front = np.array(bench_front)
# np.save('data/bench_pf/[cifar10-101][FLOPS-TEST_ERR]-108EP.npy', bench_front)
# print('Done')

# import matplotlib.pyplot as plt

# bench_front = np.load('data/bench_pf/[cifar10-101][FLOPS-TEST_ERR]-108EP.npy')

# #from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# #I  = NonDominatedSorting(method='efficient_non_dominated_sort').do(bench_front, only_non_dominated_front=True)
# # bench_front = bench_front[I]    
# # print(len(bench_front))
# plt.scatter(bench_front[:, 0], bench_front[:, 1], label='optimal front')
# plt.xlabel('Flops')
# plt.ylabel('Test err') 
# plt.grid(True, linestyle='--')
# plt.legend(loc='best')
# plt.show()

# np.save('data/bench_pf/[cifar10-101][FLOPS-TEST_ERR]-108EP.npy', bench_front)

import lib.benchmarks.nasbench301.nasbench301 as nb
from os.path import expanduser
api = nb.load_ensemble(os.path.join(expanduser('~'), '.torch/nb_models/xgb_v1.0'))
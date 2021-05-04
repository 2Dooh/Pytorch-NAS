from collections import namedtuple
from numpy.core.defchararray import upper
from pymoo.model.decision_making import normalize

# import ConfigSpace.hyperparameters as CSH


import torch
from torch.functional import norm

from utils.neural_net.flops_benchmark import get_model_infos

from lib.darts.cnn.model import NetworkCIFAR

# bench = {}
# lst = []
# optimizers = [opt for opt in os.listdir('experiments/nb301_data/nb_301_v13')]
# for optimizer in optimizers:
#     path = 'experiments/nb301_data/nb_301_v13/{}'.format(optimizer)
#     for file in os.listdir(path):
#         if 'results' in file and file.endswith('.json'):
#             with open(os.path.join(path, file), 'rb') as f:
#                 result = json.load(f)
#                 lst += [result]
#     print('Logged {}'.format(optimizer))
# # assert(len(lst) == 2243)

#     bench[optimizer] = lst.copy()
#     lst = []

import re
from functools import partial

only_numeric_fn = lambda x: int(re.sub("[^0-9]", "", x))
custom_sorted = partial(sorted, key=only_numeric_fn)
def parse_config(config, cell_type):
        cell = []

        edges = custom_sorted(
            list(
                filter(
                    re.compile('.*edge_{}*.'.format(cell_type)).match,
                    config
                )
            )
        ).__iter__()
        nodes = custom_sorted(
            list(
                filter(
                    re.compile('.*inputs_node_{}*.'.format(cell_type)).match,
                    config
                )
            )
        ).__iter__()
        op_1 = config[next(edges)]
        op_2 = config[next(edges)]
        cell.extend([(op_1, 0), (op_2, 1)])
        for node in nodes:
            op_1 = config[next(edges)]
            op_2 = config[next(edges)]
            input_1, input_2 = map(int, config[node].split('_'))
            cell.extend([(op_1, input_1), (op_2, input_2)])
        return cell

import numpy as np
bench = torch.load('experiments/bench301.pth.tar')
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
normal_bound = []
reduce_bound = []
for opt, results in bench.items():
    for i, res in enumerate(results):
        config = res['optimized_hyperparamater_config']
        base_str = 'NetworkSelectorDatasetInfo:darts:'
        normal = parse_config(config, 'normal')
        reduce = parse_config(config, 'reduce')

        a = [x[1] for x in normal]
        normal_bound += [a]
        
        b = [x[1] for x in reduce]
        reduce_bound += [b]
        # normal_concat = list(range(2, 6))
        # reduce_concat = list(range(2, 6))
        # genotype = Genotype(normal=normal, normal_concat=normal_concat, reduce=reduce, reduce_concat=reduce_concat)
        # model = NetworkCIFAR(C=config[base_str + 'init_channels'], 
        #                      num_classes=10, 
        #                      layers=config[base_str + 'layers'], 
        #                      auxiliary=config[base_str + 'auxiliary'], 
        #                      genotype=genotype).cuda()
        # model.drop_path_prob = 0.1
        # flops, params = get_model_infos(model, [1, 3, 32, 32])
        # bench[opt][i]['flops'] = flops
        # bench[opt][i]['n_params'] = params
normal_bound = np.array(normal_bound)
lwb = normal_bound.min(axis=0)
ub = normal_bound.max(axis=0)
print('normal lower bound: {}'.format(lwb))
print('normal upper bound: {}'.format(ub))

reduce_bound = np.array(reduce_bound)
lwb = reduce_bound.min(axis=0)
ub = reduce_bound.max(axis=0)
print('reduce lower bound: {}'.format(lwb))
print('reduce upper bound: {}'.format(ub))

# torch.save(bench, 'experiments/bench301.pth.tar')

# import numpy as np
# bench = torch.load('experiments/bench301_flops_params.pth.tar')
# F = []
# for opt, results in bench.items():
#     for i, res in enumerate(results):
#         test_err = 100 - res['test_accuracy']
#         flops = res['flops']
#         F += [[flops, test_err]]
#         print('[{}][{}] flops: {} - test_err: {}'.format(opt, i, flops, test_err))
#         if res['budget'] != 100:
#             print('[{}][{}] budget: {}'.format(opt, i, res['budget']))

# F = np.array(F)
# from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
# I = NonDominatedSorting(method='efficient_non_dominated_sort').do(F, only_non_dominated_front=True)
# front = F[I]
# np.save('bench_pf/[cifar10-darts][FLOPS-TEST_ERR]-100EP.npy', front)

# import numpy as np

# import matplotlib.pyplot as plt

# from pymoo.decision_making.high_tradeoff import HighTradeoffPoints

# dm = HighTradeoffPoints(epsilon=1000, normalize=True)

# front = np.load('bench_pf/[cifar10-darts][FLOPS-TEST_ERR]-100EP.npy')
# I = dm.do(front)

# plt.scatter(front[I][:, 0], front[I][:, 1], label='high trade-off points', color='green')
# plt.plot(front[:, 0], front[:, 1])
# plt.scatter(front[:, 0], front[:, 1], marker='.', color='blue', label='optimal front')
# plt.xlabel('Flops')
# plt.ylabel('Test Error')
# plt.title('Optimal Front on DARTS Space (100 epochs)')
# plt.grid(True, linestyle='--')
# plt.legend(loc='best')
# plt.show()



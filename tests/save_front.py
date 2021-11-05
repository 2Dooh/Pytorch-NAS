import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import torch

import matplotlib.pyplot as plt

from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
from pymoo.util.nds.efficient_non_dominated_sort import efficient_non_dominated_sort

def get_objective_space(obj):
    F = []
    for cost_info, more_info in obj:
        latency = cost_info['latency']
        flops = cost_info['flops']
        params = cost_info['params']
        test_err = 100 - more_info['test-accuracy']
        F += [[flops, params, latency, test_err]]

    F = np.reshape(F, (len(F), -1))
    return F

path = 'data/[cifar10-tss-200].pth.tar'
obj = torch.load(path)['obj']
F = get_objective_space(obj)
I = NonDominatedSorting(method='efficient_non_dominated_sort').do(F, only_non_dominated_front=True)
front = F[I]

# plt.scatter(front[:, 0], front[:, 1])
# plt.show()

name = 'data/bench_pf/[cifar10-tss][FLOPS-PARAMS-LATENCY-TEST_ERR]-200EP.npy'
np.save(name, front)
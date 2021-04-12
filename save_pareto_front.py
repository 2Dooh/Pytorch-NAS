import numpy as np

import torch

from utils.moeas.non_dominated_rank import non_dominated_rank

def get_objective_space(obj):
    F = []
    for cost_info, more_info in obj:
        flops = cost_info['flops']
        test_err = 100 - more_info['test-accuracy']
        F += [[flops, test_err]]

    F = np.reshape(F, (len(F), -1))
    return F

path = 'experiments/[cifar100-tss-200].pth.tar'
obj = torch.load(path)['obj']
F = get_objective_space(obj)
front = non_dominated_rank(F)

name = 'bench_pf/[cifar100-tss][FLOPS-TEST_ERROR]-200EP'
np.save(name, front)
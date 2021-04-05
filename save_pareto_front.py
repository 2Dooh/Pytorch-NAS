import numpy as np

import torch

def is_dominated(x, y):
    not_dominated = x <= y
    dominate = x < y

    return np.logical_and(
        not_dominated.all(axis=1),
        dominate.any(axis=1)
    )

def domination_count(F_pop):
        count = np.empty((F_pop.shape[0],))
        for i in range(F_pop.shape[0]):
            count[i] = is_dominated(F_pop, F_pop[i]).sum()
        return count

def non_dominated_rank(f_pop):
        count = domination_count(f_pop)

        return f_pop[count == 0]

def get_objective_space(obj):
    F = []
    for cost_info, more_info in obj:
        flops = cost_info['flops']
        test_err = 100 - more_info['test-accuracy']
        F += [[flops, test_err]]

    F = np.reshape(F, (len(F), -1))
    return F

path = 'experiments/[cifar10-sss].pth.tar'
obj = torch.load(path)['obj']
F = get_objective_space(obj)
front = non_dominated_rank(F)

name = 'bench_pf/[cifar10-sss][FLOPS-VAL_ERR]'
np.save(name, front)
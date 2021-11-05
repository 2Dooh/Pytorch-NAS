import torch

import os

import numpy as np

import re

from scipy import stats


import random

def get_runtime(f_dir, **kwargs):
    runtimes = []
    for run in sorted(os.listdir(f_dir)):
        run_dir = os.path.join(f_dir, run, 'checkpoints')
        gens = sorted(os.listdir(run_dir), key=lambda x: int(re.search(r'-(\d+)', x).group(1)))
        # assert(len(gens) == 60)
        # print(gens)
        last_gen = gens[-1]
        time = list(torch.load(os.path.join(run_dir, last_gen))['score_dict']['runtimes'].values())
        # assert(len(time) == 59)
        avg_times = [sum(t) for t in time]
        runtimes += [avg_times]

    runtimes = np.array(runtimes)
    return runtimes

def get_runtime_from_log(path, **kwargs):
    logs_dir = os.path.join(path, 'logs')
    times = []
    for file in os.listdir(logs_dir):
        if 'debug' in file:
            with open(os.path.join(logs_dir, file), 'r+') as f:
                gens = \
                    re.findall(
                        "- PopLogger - : \{'current_gen': (\d+), 'n_evals': \d+, 'time': (\d+.\d+)}",                             f.read()
                    )
            gens = [(int(gen), float(time)) for gen, time in gens]
            times += [*gens]
    times = sorted(times, key=lambda x: x[0])
    # assert(len(times) == 60)
    return [time for _, time in times][:100]    

def get_log_runtimes_from_runs():
    runtimes = []
    path = 'experiments/101-FLOPS-NTK-LR'
    for run in sorted(os.listdir(path)):
        runtimes += [get_runtime_from_log(os.path.join(path, run))]
    runtimes = np.array(runtimes)
    # assert(runtimes.shape == (29, 60))
    return runtimes

# runtimes = get_log_runtimes_from_runs().mean(axis=0)
# runtimes = get_runtime('experiments/101-flops-valid').mean(axis=0)

baseline = torch.load('data/baseline101.pth.tar')
gf = torch.load('data/101-new-temp.pth.tar')

idx = list(set(list(range(30))) - set([14, 15, 16, 17, 22, 23, 28, 29]))

F_baseline = [res[2] for res in baseline]
F_gf = [res[2] for res in gf]

baseline_best = np.array([run[-1][:, 1].min() for run in F_baseline])
gf_best = np.array([run[-1][:, 1].min() for run in F_gf])
print('baseline:')
print('best: {}'.format(100 - baseline_best.min()))
print(np.unique(baseline_best, return_counts=True))
print('Avg: {:.3f} - Std: {:.3f}\n\n'.format(100 - baseline_best.mean(), baseline_best.std()))

print('gf:')
print('best: {}'.format(100 - gf_best.min()))
print(np.unique(gf_best, return_counts=True))
print('Avg: {:.3f} - Std: {:.3f}'.format(100 - gf_best.mean(), gf_best.std()))

# bench101 = np.load('data/bench_pf/[cifar10-101][FLOPS-TEST_ERR]-108EP.npy')
# print('Optimal 101: {}'.format(100 - bench101.min()))

# print('Debug')
# print('Num better: {}'.format(sum(gf_best < baseline)))

print('Num better: {}'.format(sum(gf_best <= baseline_best)))
print('T-Test res: ')
print(stats.ttest_ind(baseline_best, gf_best))
# print(stats.tt)
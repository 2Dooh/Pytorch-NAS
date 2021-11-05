from pickle import load
from nats_bench import create

import torch

import numpy as np

from pymoo.factory import get_performance_indicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import re

import matplotlib.pyplot as plt

import lib.models.cell_operations as OPS

import os

from scipy import stats


api = create(None, 'tss', fast_mode=True, verbose=False)
benchmark = torch.load('data/[ImageNet16-120-tss-200].pth.tar')['obj']
pf = np.load('data/bench_pf/[ImageNet16-120-tss][FLOPS-TEST_ERR]-200EP.npy')

metric = get_performance_indicator('igd', pf=pf, normalize=True)

nodes = [0, 0, 1, 0, 1, 2]
primitives = np.array(OPS.NAS_BENCH_201)
def bench201_decoder(x):
    ops = primitives[x]
    strings = ['|']

    for i, op in enumerate(ops):
        strings.append(op+'~{}|'.format(nodes[i]))
        if i < len(nodes) - 1 \
            and nodes[i+1] == 0:
            strings.append('+|')
    return ''.join(strings)

def bench201_query(genotype, **kwargs):
    idx = api.query_index_by_arch(genotype)

    flops = benchmark[idx][0]['flops']
    test_err = \
        100 - benchmark[idx][1]['test-accuracy']
    return [flops, test_err]


def convert_2_obj_space(checkpoint_path,
                        decoder,
                        func,
                        metric,
                        return_non_dominated=True,
                        convert_obj=True,
                        **kwargs):
    max_r = 0
    n_evals, F, cv = [], [], []
    checkpoints = sorted(os.listdir(checkpoint_path), 
                         key=lambda x: int(re.search(r'-(\d+)', x).group(1)))
    for i, filename in enumerate(checkpoints):
        checkpoint = torch.load(os.path.join(checkpoint_path, filename))
        obj = checkpoint['obj']
        n_evals += [obj.evaluator.n_eval]
        opt = obj.opt
        feas = np.where(opt.get('feasible'))[0]
        X = checkpoint['elitist_archive'].archive['X']
        _F = checkpoint['elitist_archive'].archive['F']
        max_r = max(max_r, len(X))
        __F = []
        if convert_obj:
            for x in X:
                genotype = decoder(x)
                f = func(genotype)
                __F += [f]
            __F = np.array(__F)
            if return_non_dominated:
                I = NonDominatedSorting().do(__F, only_non_dominated_front=True)
                F += [__F[I]]
            else:
                F += [__F]
        else:
            F += [_F]
    score = [metric.calc(f) for f in F]
    print('max_archive_size: {}'.format(max_r))
    return n_evals, score, F

def load_runs(f_dir, name, save=False, **kwargs):
    data = []
    for run in sorted(os.listdir(f_dir)):
        res = convert_2_obj_space(os.path.join(f_dir, run, 'checkpoints'), **kwargs)
        data += [res]
    if save:
        torch.save(data, 'data/{}.pth.tar'.format(name))
    return data

# baseline = load_runs(
#     'experiments/TSS-FLOPS-VALID', 
#     'baseline201_to_imagenet', 
#     save=True, 
#     decoder=bench201_decoder, 
#     func=bench201_query,
#     metric=metric
# )

# gf = load_runs(
#     'experiments/TSS-FLOPS-NTK-LR', 
#     'gf201_to_imagenet', 
#     save=True, 
#     decoder=bench201_decoder, 
#     func=bench201_query,
#     metric=metric
# )

baseline = torch.load('data/baseline201_to_imagenet.pth.tar')
gf = torch.load('data/gf201_to_imagenet.pth.tar')

score_baseline = []; score_gf = []

for i in range(29):
    score_baseline += [baseline[i][1][-1]]
    score_gf += [gf[i][1][-1]]


score_baseline = np.array(score_baseline)
score_gf = np.array(score_gf)


print('Baseline: mean - {} - std - {}'.format(score_baseline.mean(), score_baseline.std()))
print('GF: mean - {} - std - {}'.format(score_gf.mean(), score_gf.std()))

print('T-Test res: ')
print(stats.ttest_ind(score_baseline, score_gf))

# score_baseline, score_gf = [], []
# for i in range(29):
#     score_baseline += [metric.calc(baseline[i][2][-1])]
#     score_gf += [metric.calc(gf[i][2][-1])]

# score_baseline = np.array(score_baseline)
# score_gf = 


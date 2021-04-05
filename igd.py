import matplotlib.pyplot as plt

from pymoo.performance_indicator.igd import IGD

import numpy as np

import torch

path = 'bench_pf/[cifar10-tss][FLOPS-VAL_ERR].npy'
pf = np.load(path)

result = torch.load('experiments/TSS201-FLOPS_VAL_ERR-UX-G200-CIFAR10/out/result.pth.tar')

print('Debug')
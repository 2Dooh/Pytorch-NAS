from nats_bench import create

import torch

import numpy as np

dataset = 'cifar10'
hp = '200'
api = create(None, 'tss', fast_mode=True, verbose=False)
F = []
for i, arch_str in enumerate(api):
    flops = api.get_cost_info(i, dataset, hp=hp)['flops']
    test_err = 100 - api.get_more_info(i, dataset, hp=hp, is_random=False)['test-accuracy']
    F += [flops, test_err]

F = np.array(F).reshape((len(api), -1))

torch.save({'flops_test-err': F}, './cifar10_tss_flops_test-err.pth.tar')

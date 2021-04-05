from nats_bench import create

import torch

import numpy as np

dataset = 'ImageNet16-120-tss'
hp = '90'
api = create(None, 'sss', fast_mode=True, verbose=False)
F = []
for i, arch_str in enumerate(api):
    cost_info = api.get_cost_info(i, dataset, hp=hp)
    more_info = api.get_more_info(i, dataset, hp=hp, is_random=False)
    F += [[cost_info, more_info]]
    print(i)

F = np.array(F).reshape((len(api), -1))

torch.save({'obj': F}, 'experiments/[ImageNet16-120-sss].pth.tar')

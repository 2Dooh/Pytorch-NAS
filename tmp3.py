import torch

import numpy as np

from nats_bench import create

api = create(None, 'tss', fast_mode=True, verbose=False)

gf = torch.load('data/gf201.pth.tar')
F_gf = [res[2] for res in gf]
gf_best = max(100 - np.array([run[-1][:, 1].min() for run in F_gf]))
benchmark = torch.load('data/[cifar10-tss-200].pth.tar')['obj']

best_idx = None
for i in range(len(benchmark)):
    if benchmark[i][1]['test-accuracy'] == gf_best:
        best_idx = i
        print(i)
        print(benchmark[i][0]['flops'])
        break

print(api.query_by_index(best_idx))
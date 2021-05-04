import torch

from nats_bench import create

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# bench_info = torch.load('experiments/[cifar10-tss-200].pth.tar')['obj']
# test_err = []
# for idx in range(len(bench_info)):
#     test_acc = bench_info[idx][1]['test-accuracy']
#     test_err += [100 - test_acc]

ntk_lr = torch.load('experiments/[cifar10-tss][NTK_LR].pth.tar')

for key in ntk_lr.keys():
    ntk_lr[key]['ntks'] = np.mean(ntk_lr[key]['ntks'])
    if ntk_lr[key]['ntks'] == 1e5:
        ntk_lr[key]['ntks'] = 1e7
    ntk_lr[key]['lrs'] = np.mean(ntk_lr[key]['lrs'])
dataset = pd.DataFrame.from_dict(ntk_lr).T
# dataset.values[:, 0] = StandardScaler().fit_transform(np.log(dataset.values[:, 0]).reshape(-1, 1)).ravel()
# dataset.values[:, 1] = StandardScaler().fit_transform(dataset.values[:, 1].reshape(-1, 1)).ravel()
# dataset['test_err'] = test_err 
print(dataset)

# plt.scatter(dataset.values[:, 0], dataset.values[:, 1])
# plt.show()

# dataset.to_csv('logs/ntk_lr_test-err.csv')

# print('ntk: mean={} std={}'.format(dataset.values[:, 0].mean(), dataset.values[:, 0].std()))
# print('lr: mean={} std={}'.format(dataset.values[:, 1].mean(), dataset.values[:, 1].std()))

# fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
fig, axs = plt.subplots()
axs.hist(dataset.values[:, 0], bins=100)
# axs[1].hist(dataset.values[:, 1]**0.5, bins=100)
plt.show()
# # print(dataset)
# print('Debug')
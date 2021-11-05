import torch

from nats_bench import create

import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

plt.rcParams["font.family"] = "serif"

bench_info = torch.load('data/[cifar100-tss-200].pth.tar')['obj']
test_acc = []
for idx in range(len(bench_info)):
    test_acc += [bench_info[idx][1]['test-accuracy']]
    # test_err += [100 - test_acc]

ntk_lr = torch.load('data/[cifar10-tss][NTK_LR].pth.tar')

max_ntk = 0
for key in ntk_lr.keys():
    ntk_lr[key]['ntks'] = np.mean(ntk_lr[key]['ntks'])
    max_ntk = max(max_ntk, ntk_lr[key]['ntks'])
    if ntk_lr[key]['ntks'] == 1e5:
        # print(ntk_lr[key]['ntks'])
        ntk_lr[key]['ntks'] = 1e7
    ntk_lr[key]['lrs'] = np.mean(ntk_lr[key]['lrs'])
print('Max NTK: {}'.format(max_ntk))
dataset = pd.DataFrame.from_dict(ntk_lr).T
data_normed = dataset.copy()
data_normed.values[:, 0] = StandardScaler().fit_transform(np.log(dataset.values[:, 0]).reshape(-1, 1)).ravel()
data_normed.values[:, 1] = StandardScaler().fit_transform(dataset.values[:, 1].reshape(-1, 1)).ravel()
# dataset['test_err'] = test_err 
# print(dataset)

# plt.scatter(dataset.values[:, 0], dataset.values[:, 1])
# plt.show()

# dataset.to_csv('logs/ntk_lr_test-err.csv')

# print('ntk: mean={} std={}'.format(dataset.values[:, 0].mean(), dataset.values[:, 0].std()))
# print('lr: mean={} std={}'.format(dataset.values[:, 1].mean(), dataset.values[:, 1].std()))

# fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
fig, axs = plt.subplots(2, 2, tight_layout=True)
axs[0, 1].scatter(dataset.values[:, 0], test_acc, marker='.')
axs[0, 1].set_xlabel(r'NTK $\kappa_{\mathcal{N}}=\frac{\lambda_{max}}{\lambda_{min}}$')
axs[0, 1].set_ylabel('CIFAR-10 Test-Acc (%)')
axs[0, 1].grid(linestyle='--')
axs[1, 1].scatter(data_normed.values[:, 0], test_acc, marker='.')
axs[1, 1].grid(linestyle='--')
axs[1, 1].set_ylabel('CIFAR-10 Test-Acc (%)')
axs[1, 1].set_xlabel(r'NTK $\kappa_{\mathcal{N}}=\frac{\lambda_{max}}{\lambda_{min}}$')

sns.histplot(
    dataset.values[:, 0],
    bins=int(np.sqrt(len(dataset))),
    kde=True,
    # hist=True,
    # hist_kws={'edgecolor':'black'},
    ax=axs[0, 0]
)
axs[0, 0].grid(linestyle='-')

sns.histplot(
    data_normed.values[:, 0],
    bins=int(np.sqrt(len(data_normed))),
    kde=True,
    ax=axs[1, 0]
)
axs[1, 0].grid(linestyle='-')
# plt.title('After Log Transformation')
# Adjust vertical_spacing = 0.5 * axes_height
plt.subplots_adjust(hspace=0.5)
plt.figtext(0.55, 0.5, 'After Log Transformation', ha='center', va='center')
plt.savefig('assets/ntk_norm.pdf')
plt.show()
# # print(dataset)
# print('Debug')
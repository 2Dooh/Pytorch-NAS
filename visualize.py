import matplotlib.pyplot as plt

import torch

import os

from pymoo.util.nds.non_dominated_sorting import find_non_dominated

import numpy as np

import re

from pprint import pprint

ckp_path = 'experiments/checkpoints'
ckps = sorted(os.listdir(ckp_path), 
                  key=lambda x: int(re.search(r'-(\d+)', x).group(1)))
F = []
X = []
for i, gen in enumerate(ckps):
    gen = torch.load(os.path.join(ckp_path, gen))
    f = gen['obj'].pop.get('F')
    x = gen['obj'].opt.get('X')
    F += [f]
    X += [x]

final_front = X[-1]

g_x = lambda x: [(2**(n_nodes + 4), 0.) for n_nodes in x if n_nodes > 0]
G_x = [g_x(x) for x in final_front]

pprint(G_x)



# os.makedirs('assets/gifs', exist_ok=True)
# # plt.plot(
# #     F[0][:, 0][np.argsort(F[0][:, 0])],
# #     F[0][:, 1][np.argsort(F[0][:, 0])],
# #     '-.o',
# #     color='red',
# #     mfc='none',
# #     label='initial'
# # )
# plt.plot(
#     F[-1][:, 0][np.argsort(F[-1][:, 0])],
#     F[-1][:, 1][np.argsort(F[-1][:, 0])],
#     '--s',
#     color='blue',
#     mfc='none',
#     label='final'
# )
# plt.plot(F[-1][2, 0], F[-1][2, 1], 'rx', label='chosen architecture')
# plt.annotate('{:.2f}, {:.2f}'.format(F[-1][2, 0], F[-1][2, 1]), (F[-1][2, 0], F[-1][2, 1] + 5))
# plt.grid(True, linestyle='--')
# # plt.title('Evolutionary progress on trade-off front')
# plt.xlabel('Trainability (Lower is better)')
# plt.ylabel('Expressivity (Lower is better)')
# plt.legend(loc='best')
# plt.show()
# xlim = (F[-1][:, 0].min() - 1000, F[0][:, 0].max() + 1000)
# ylim = (F[-1][:, 1].min() - 10, F[0][:, 1].max() + 10)
# xmin, xmax = (np.inf, -np.inf)
# ymin, ymax = (np.inf, -np.inf)
# for i, f in enumerate(F):
#     xmin = min(f[:, 0].min() - 1000, xmin)
#     xmax = max(f[:, 0].max() + 1000, xmax)
#     ymin = min(f[:, 1].min() - 10, ymin)
#     ymax = max(f[:, 1].max() + 10, ymax)
# for i, f in enumerate(F):
#     # I = find_non_dominated(f, f)
#     # indices = list(range(len(f)))
#     # mask = list(set(indices) - set(I))
#     # plt.scatter(f[mask, 0], f[mask, 1], color='blue', mfc='none')
#     plt.plot(
#         f[:, 0][np.argsort(f[:, 0])], 
#         f[:, 1][np.argsort(f[:, 0])], 
#         's--', color='red', mfc='none', label='gen {}'.format(i+1)
#     )

#     plt.grid(True, linestyle='--')
#     plt.title('Evolutionary progress on trade-off front')
#     plt.xlabel('Trainability (Lower is better)')
#     plt.ylabel('Expressivity (Lower is better)')
#     plt.legend(loc='best')
#     plt.xlim([xmin, xmax])
#     plt.ylim([ymin, ymax])
#     plt.savefig('assets/gifs/g{:0>3d}.png'.format(i+1))
#     plt.cla()




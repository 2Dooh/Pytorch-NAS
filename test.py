import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pymoo.model.decision_making import normalize
from torch.functional import norm
plt.rcParams["font.family"] = "serif"

import torch

import numpy as np


from pymoo_custom_modules.problems.multi_obj.bench201 import SSSBench201, TSSBench201
from pymoo.factory import get_decision_making
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting, find_non_dominated
from pymoo.model.decision_making import normalize
from pymoo.visualization.scatter import Scatter
# init = torch.load('experiments/exp-20210323-1754/checkpoints/[NSGAII] Gen_1.pth.tar')
# last = torch.load('experiments/[NSGAII] Gen_40.pth.tar')
# # f1 = gen1['eval_dict']['f_pop_obj']
# f_last = last['eval_dict']['f_pop_obj']
# current_gen = last['gen']
# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(f_last[:, 0], f_last[:, 1], f_last[:, 2])
# # ax.scatter(f1[:, 0], f1[:, 1], f1[:, 2], color='red')

# ax.set_xlabel('FLOPS')
# ax.set_ylabel('NTK')
# ax.set_zlabel('LR')
# plt.show()
# print('debug')

# def get_err(X, problem, hp):
#     indices = []
#     flops, t_err_X, v_err_X, v_err_X_max = [], [], [], []
#     for i in range(X.shape[0]):
#         x = X[i]
#         p_x = problem._decode(x)
#         idx_x = problem.api.query_index_by_arch(p_x)
#         indices += [idx_x]
#     # indices = list(set(indices))
#     # for idx_x in indices:
#         flop = problem.api.get_cost_info(idx_x, 'cifar10', hp=hp)['flops']
#         flops += [flop]
#         t_err = \
#             100 - problem.api.get_more_info(idx_x, 'cifar10', hp=hp, is_random=False)['test-accuracy']
#         v_err = \
#             100 - problem.api.get_more_info(idx_x, 
#                                             dataset+'-valid' if dataset == 'cifar10' else dataset, 
#                                             iepoch=24, 
#                                             hp=hp, 
#                                             is_random=False)['{}-accuracy'.format('valid' if dataset == 'cifar10' else 'valtest')]
#         v_err_max = \
#             100 - problem.api.get_more_info(idx_x, dataset, hp=hp, is_random=False)['train-accuracy']
#         v_err_X += [v_err]
#         v_err_X_max += [v_err_max]
#         t_err_X += [t_err] 
def get_err(X, problem):
    __F = []
    for x in X:
            p_x = problem._decode(x)
            idx_x = problem.api.query_index_by_arch(p_x)
            # flops = problem.api.get_cost_info(idx_x, DATASET, hp='200')['flops']
            flops = bench_info[idx_x][0]['flops']
            test_err = 100 - bench_info[idx_x][1]['test-accuracy']
            # test_err = \
            #     100 - problem.api.get_more_info(idx_x, 
            #                                     DATASET, 
            #                                     hp=hp, 
            #                                     is_random=False)['test-accuracy']
            __F += [[flops, test_err]]
    # t_err_X = list(set(t_err_X))
    # flops = list(set(flops))
    # v_err_X = list(set(v_err_X))
    __F = np.array(__F)
    return __F

def plot(X, Y, color, linestyle, label):
    ax.plot(X, Y, color=color, linestyle=linestyle, label=label)
def scatter(X, Y, marker, color):
    ax.scatter(X, Y, color=color, marker=marker)

def plot_front(X, 
               Y,
               marker, 
               linestyle, 
               color, 
               label):
    sorted_idx = np.argsort(X)
    plot(X[sorted_idx], Y[sorted_idx], color, linestyle, label)
    scatter(X[sorted_idx], Y[sorted_idx], marker, color)

bench_info = torch.load('experiments/[cifar10-tss-200].pth.tar')['obj']

dataset = 'cifar10'
fig, ax = plt.subplots()

tss_bench = TSSBench201(dataset)
sss_bench = SSSBench201(dataset)

# A = torch.load('experiments/[NSGAII] Gen_200.pth_11.tar')
A = torch.load('experiments/[TSS-CIFAR10][FLOPS_VALID]-UX-G200-S777-25EP/checkpoints/[NSGA2_TSSBench201] G-200.pth.tar')
B = torch.load('experiments/ntk_lr_prune_worst/[NSGA2_TSSBench201GradientFree] G-200.pth.tar')
pf = np.load('bench_pf/[cifar10-tss][FLOPS-TEST_ERR]-200EP.npy')
from pymoo.decision_making.high_tradeoff import HighTradeoffPoints

# dm = get_decision_making('high-tradeoff', normalize=False)
dm = HighTradeoffPoints(epsilon=0.75, normalize=False)
#pf = normalize(pf)
I = dm.do(pf)

# plot = Scatter()
# plot.add(pf, alpha=0.2)
# plot.add(pf[I], color="red", s=100)
# plot.show()
# F = []
# for cost_info, more_info in front['obj']:
#     flops = cost_info['flops']
#     test_err = 100 - more_info['test-accuracy']
#     F += [[flops, test_err]]

# F = np.array(F).reshape((len(F), -1))
# front = non_dominated_rank(F)
# plot_front(B['eval_dict']['f_pop_obj'][:, 0],
#            B['eval_dict']['f_pop_obj'][:, 1],
#            marker='x',
#            linestyle='--',
#            color='blue',
#            label='FLOPS - NTK_LR (200)')

# plot_front(A['eval_dict']['f_pop_obj'][:, 0],
#            A['eval_dict']['f_pop_obj'][:, 1],
#            marker='x',
#            linestyle='--',
#            color='red',
#            label='FLOPS - NTK_LR (40)')

F_A = get_err(A['obj'].opt.get('X'), tss_bench)
F_B = get_err(B['elitist_archive']['X'], tss_bench)

# IA = NonDominatedSorting().do(F_A, only_non_dominated_front=True)
# IB = NonDominatedSorting().do(F_B, only_non_dominated_front=True)

IA = dm.do(F_A)
IB = dm.do(F_B)
# plot_front(flops_B, v_err_B, 
#            marker='x', 
#            linestyle='--', 
#            color='green',
#            label='(G:{}) Test Err 25 epochs [FLOPS-NTK-LR]'.format(B['obj'].n_gen))
# plot_front(flops_A, v_err_A, 
#            marker='o', 
#            linestyle='--', 
#            color='blue',
#            label='(G:{}) Test Err 25 epochs [FLOPS-VALID_ERR]'.format(A['obj'].n_gen))

plot_front(F_B[IB][:, 0], F_B[IB][:, 1], 
           marker='v', 
           linestyle='--', 
           color='blue',
           label='(G:{}) Test Err 200 epochs [FLOPS-NTK-LR]'.format(B['obj'].n_gen))
plot_front(F_A[IA][:, 0], F_A[IA][:, 1], 
           marker='^', 
           linestyle='--', 
           color='red',
           label='(G:{}) Test Err 200 epochs [FLOPS-VALID_ERR]'.format(A['obj'].n_gen))

# plot_front(flops_A, v_err_A_max, 
#            marker='o', 
#            linestyle='-', 
#            color='purple',
#            label='(G:{}) Val Err 200 epochs [FLOPS-VALID_ERR]'.format(A['obj'].n_gen))

# plot_front(flops_B, v_err_B_max, 
#            marker='x', 
#            linestyle='-', 
#            color='olive',
#            label='(G:{}) Val Err 200 epochs [FLOPS-NTK-LR]'.format(B['obj'].n_gen))

plot_front(pf[:, 0], pf[:, 1], 
           marker='*', 
           linestyle='-', 
           color='black',
           label='Pareto Front')
ax.scatter(pf[I][:, 0], pf[I][:, 1],
          marker='o',
          color='purple',
          s=50,
          label='high trade-off')

ax.set_xlabel('FLOPS')
ax.set_ylabel('Error Rate (%)')
# ax.set_ylabel('NTK_LR')
ax.legend(loc='best')
ax.grid(True, linestyle='--')
ax.set_title(dataset.upper() + ' 200 epochs')
plt.show()
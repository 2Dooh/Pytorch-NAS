import matplotlib.pyplot as plt

from pymoo.performance_indicator.igd import IGD

import numpy as np

import torch

import os

import re

from procedures.problems.bi_obj.bench201 import SSSBench201, TSSBench201

from pymoo.util.nds.non_dominated_sorting import find_non_dominated





search_space = {'sss': SSSBench201, 'tss': TSSBench201}

path = 'bench_pf/[cifar10-tss][FLOPS-TEST_ERR]-200EP.npy'
pf = np.load(path)

bench_info = torch.load('experiments/[cifar10-tss-200].pth.tar')['obj']

DATASET, SS, F1, F2, HP = re.findall(r'.+\/\[(\w+)-(\w+)\]\[(\w+)-(\w+)]-(\d+).+', path)[0]

problem = search_space[SS](dataset=DATASET)

def get_result(checkpoint_path, key, elitist_archive=True):
    n_evals, F, cv = [], [], []
    checkpoints = sorted(os.listdir(checkpoint_path), key=key)
    pf = None
    c_eval = []
    max_r = 0
    for i, checkpoint in enumerate(checkpoints):
        checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint))
        obj = checkpoint['obj']
        n_evals += [obj.evaluator.n_eval]
        opt = obj.opt
        feas = np.where(opt.get("feasible"))[0]
        # cv += [opt.get('CV').min()]
        # _F = opt.get('F')[feas]
        # X = opt.get('X')[feas]
        if elitist_archive:
            if isinstance(checkpoint['elitist_archive'], dict):
                X = checkpoint['elitist_archive']['X']
                _F = checkpoint['elitist_archive']['F']
            else:
                X = checkpoint['elitist_archive'].archive['X']
                _F = checkpoint['elitist_archive'].archive['F']
        else:
            X = opt.get('X')[feas]
            _F = opt.get('F')[feas]
            # _, I = np.unique(X, axis=0, return_index=True)
            # X = X[I]
            # _F = _F[I]

        
        max_r = max(max_r, len(X))
        pf = opt.get('F')

        __F = []
        for x in X:
            if x.dtype == np.bool_:
                b2i = lambda a: int(''.join(str(int(bit)) for bit in a), 2)
                x = [b2i(x[start:start+3]) for start in np.arange(x.shape[0])[::3]]
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
        __F = np.array(__F)
        # _F = np.column_stack((_F[:, 0], __F))

        # assert((_F[:, 0] == __F[:, 0]).all()) 
        I = find_non_dominated(__F, __F)
        # __F, _ = non_dominated_rank(__F)
        F += [__F[I]]


        # temp = F_all.sum(axis=1)
        # temp = F[i].sum(axis=1)
        # temp = F_all.sum(axis=1) == F[i].sum(axis=1)
        # candidate = F_all[np.where(F_all.sum(axis=1) == F[i].sum(axis=1))[0]]
        # if len(candidate) == 0:
        #     n_evals.pop(i)
        #     continue
        # elistic_archive[i] = candidate
        # F_all = F_all[F_all != F[i]]

    # for i in range(len(F)):
    #     dominated_mask = np.zeros((F[i].shape[0])).astype(np.bool)
    #     for j, f in enumerate(F[i]):
    #         dominated = is_dominated(F_all, f)
    #         dominated_mask[j] = True in dominated
    #     F[i] = F[i][dominated_mask == False]
    #     if len(F[i]) == 0:
    #         n_evals.pop(i)
    print(max_r)
    return n_evals, F, cv, pf

# F_1 = get_result('experiments/[NSGA2-TSS-CIFAR10][FLOPS_TEST]-UX-G200-S0 (ELIMINATE-DUP) (NO RE-EVAL)/checkpoints',
#                  lambda x: int(re.search(r'-(\d+)', x).group(1)),
#                  hp=HP)

# F_2 = get_result('experiments/rep-3-new-g30/checkpoints',
#                  lambda x: int(re.search(r'-(\d+)', x).group(1)),
#                  hp=HP)
# F_3 = get_result('experiments/rank_no_re-eval',
#                  lambda x: int(re.search(r'-(\d+)', x).group(1)),
#                  hp=HP)

# F_4 = get_result('experiments/[NSGA2-TSS-CIFAR10][FLOPS_VALID]-UX-G200-S777-25EP/checkpoints',
#                  lambda x: int(re.search(r'-(\d+)', x).group(1)),
#                  hp=HP)
# F_1 = get_result('experiments/norm_g8/checkpoints',
#                  lambda x: int(re.search(r'-(\d+)', x).group(1)),
#                  hp=HP)

# F_5 = get_result('experiments/norm_zero_no_re-eval',
#                  lambda x: int(re.search(r'-(\d+)', x).group(1)),
#                  hp=HP)


metric = IGD(pf=pf, normalize=True)
# igd_1 = [metric.calc(f) for f in F_1[1] if len(f) != 0]
# igd_2 = [metric.calc(f) for f in F_2[1] if len(f) != 0]
# igd_3 = [metric.calc(f) for f in F_3[1] if len(f) != 0]
# igd_4 = [metric.calc(f) for f in F_4[1] if len(f) != 0]
# igd_5 = [metric.calc(f) for f in F_5[1] if len(f) != 0]
# # visualize the convergence curve 
# plt.plot(F_1[0], igd_1, '-', markersize=4, linewidth=1, color="green", label='Flops - NTK + LR (Normed) (1)')
# plt.plot(F_4[0], igd_4, '-', markersize=4, linewidth=1, color="blue", label='Flops - Valid Err')
# plt.plot(F_2[0], igd_2, '-', markersize=4, linewidth=1, color="red", label='Flops - NTK - LR')


# x_new = np.linspace(min(F_3[0]), max(F_3[0]), 5000)
# spl = make_interp_spline(F_3[0], igd_3, k=3)
# igd_3_smooth = spl(x_new)

# plt.plot(F_3[0], igd_3, '-', markersize=4, linewidth=1, color="black", label='Flops - NTK+LR Rank (Allow Dup)')
# plt.plot(F_5[0], igd_5, '-', markersize=4, linewidth=1, color="magenta", label='Flops - NTK + LR (Normed) (2)')
# # plt.yscale("log")          # enable log scale if desired
# plt.xscale('log')
# plt.suptitle('Objective Space: {} - VALID-ERR ({} epochs)'.format(F1, 25))
# plt.title("Convergence")
# # plt.title("Convergence to Pareto Front (AVG Test-Error of {} EP)".format(HP))
# plt.xlabel("Function Evaluations")
# plt.ylabel("IGD")
# plt.legend(loc='best')
# plt.grid(True, linestyle='--')
# # plt.savefig('assets/{}_Val-Err_{}-own_front-{}-s777-log_x.pdf'.format(SS.upper(), F1, 25))
# plt.show()


F_4 = get_result('experiments/[TSS-CIFAR10][FLOPS_VALID]-UX-G200-S777-200EP-NO_ISO/checkpoints',
                    lambda x: int(re.search(r'-(\d+)', x).group(1)), elitist_archive=True)

# F_5 = get_result('experiments/norm_zero_no_re-eval',
#                     lambda x: int(re.search(r'-(\d+)', x).group(1)),
#                     hp=HP)

# F_3 = get_result('experiments/good_run/ntk_lr_mean',
#                     lambda x: int(re.search(r'-(\d+)', x).group(1)))


# igd_3 = [metric.calc(f) for f in F_3[1] if len(f) != 0]
igd_4 = [metric.calc(f) for f in F_4[1]]


stop = False
while not stop:
    try:
    # F_2 = get_result('experiments/3-obj-fixed',
    #             lambda x: int(re.search(r'-(\d+)', x).group(1)),
    #             hp=HP)
    # F_6 = get_result('experiments/norm_zero_no_dup',
    #                 lambda x: int(re.search(r'-(\d+)', x).group(1)),
    #                 hp=HP)
        F_5 = get_result('experiments/[TSS-CIFAR10][FLOPS_VALID]-UX-G200-S777-200EP-INT/checkpoints',
                        lambda x: int(re.search(r'-(\d+)', x).group(1)),
                        elitist_archive=True)
    # F_1 = get_result('experiments/ntk_lr_prune_worst',
    #                 lambda x: int(re.search(r'-(\d+)', x).group(1)))
    # igd_1 = np.array([metric.calc(f) for f in F_1[1]])
        igd_5 = np.array([metric.calc(f) for f in F_5[1]])

    # assert((igd_1 != igd_5).any())
    # igd_2 = [metric.calc(f) for f in F_2[1] if len(f) != 0]
    # igd_6 = [metric.calc(f) for f in F_6[1] if len(f) != 0]


    # plt.plot(F_1[0], igd_1, '-', markersize=4, linewidth=1, color="green", label='F - NTK - LR (Elitist archive)')
        plt.plot(F_4[0], igd_4, '-', markersize=4, linewidth=1, color="blue", label='F - Val_Err (Binary)')
        # plt.plot(F_2[0], igd_2, '-', markersize=4, linewidth=1, color="red", label='F - NTK - LR')

        # plt.plot(F_3[0], igd_3, '-', markersize=4, linewidth=1, color="black", label='F - NTK - LR Mean')
        plt.plot(F_5[0], igd_5, '-', markersize=4, linewidth=1, color="magenta", label='F - Val_Err (Int)')
        # plt.plot(F_6[0], igd_6, '-', markersize=4, linewidth=1, color="purple", label='F - NTK+LR Normed (Dup Remove)')
        # plt.plot(F_7[0], igd_7, '-', markersize=4, linewidth=1, color="gray", label='F - NTK+LR Rank (Dup Remove)')

        plt.xscale('log')
        plt.suptitle('Objective Space: {} - VALID-ERR ({} epochs)'.format(F1, 25))
        # plt.title("Convergence")
        plt.title("Convergence to Pareto Front (AVG Test-Error of {} EP)".format(HP))
        plt.xlabel("Function Evaluations")
        plt.ylabel("IGD")
        plt.legend(loc='best')
        plt.grid(True, linestyle='--')
        plt.pause(0.1)
        plt.cla()
    except KeyboardInterrupt:
        stop = True
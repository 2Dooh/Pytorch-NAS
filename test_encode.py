import numpy as np


n_inter_nodes = 4
n_input_nodes = 2
n_ops = 7

ops = list(range(n_ops))
print(list(range(2+3)))

xl, xu = [], []
for n in range(n_inter_nodes):
    xu += [len(list(range(n_input_nodes+n))) - 1]

xu = np.repeat(xu, repeats=n_input_nodes)

xl = np.zeros_like(xu)
xl[1::2] += 1; xu[::n_input_nodes] -= 1    # prevent 2nd input to block the same as the 1st one

X = np.ones(32) * 6
X[1::2] = ...
print(xl)
print(xu)

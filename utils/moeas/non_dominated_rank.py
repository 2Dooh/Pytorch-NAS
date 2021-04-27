import numpy as np

def is_dominated(x, y):
    not_dominated = x <= y
    dominate = x < y

    return np.logical_and(
        not_dominated.all(axis=1),
        dominate.any(axis=1)
    )

def domination_count(F_pop):
        count = np.empty((F_pop.shape[0],))
        for i in range(F_pop.shape[0]):
            count[i] = is_dominated(F_pop, F_pop[i]).sum()
        return count

def non_dominated_rank(f_pop):
        indices = np.arange(len(f_pop))
        count = domination_count(f_pop)
        return f_pop[count == 0], indices[count == 0]
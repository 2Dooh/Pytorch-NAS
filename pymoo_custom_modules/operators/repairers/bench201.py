import numpy as np

from pymoo.model.repair import Repair

class Bench201Repairer(Repair):
    def _do(self, problem, pop, **kwargs):
        Z = pop.get('X')
        for i in range(Z.shape[0]):
            x = Z[i]
            for start in np.arange(x.shape[0])[::3]:
                y = x[start:start+3]
                if y[0] == 1 and y.sum() > 1:
                    Z[i][start] = 0

        pop.set('X', Z)
        return pop
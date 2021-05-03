import numpy as np

from pymoo.model.repair import Repair
from pymoo.util.normalization import denormalize

class Bench301Repairer(Repair):
    def _do(self, problem, pop, **kwargs):
        Z = pop.get('X')

        # edge_connections = Z[:, 1::2]
        # for prev_prev in np.arange(edge_connections.shape[1])[::2]:
        #     prev = prev_prev + 1
            
        #     if edge_connections[prev_prev] == edge_connections[prev]:
        #         edge_connections[prev] += 1

        for i, z in enumerate(Z):
            edge_connections = z[1::2]
            for prev_prev in np.arange(len(edge_connections))[::2]:
                prev = prev_prev + 1

                if edge_connections[prev_prev] == edge_connections[prev]:
                    edge_connections[prev] += 1
            z[1::2] = edge_connections
            Z[i] = z
        

        pop.set('X', Z)
        return pop
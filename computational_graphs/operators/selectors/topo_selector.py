import computational_graphs.operators.base as base

from itertools import cycle

import torch

class TopoSelector(base.OperatorBase):
    def __init__(self, topology='star', **kwargs):
        super().__init__(**kwargs)
        self.topology = topology
        if topology not in ['star', 'ring']:
            self.logger.error('star or ring topology only!')

    def __call__(self, **kwargs):
        if self.topology == 'star':
            return self.__star_select(**kwargs)
        else:
            return self.__ring_select(**kwargs)

    def __ring_select(self, f_pop):
        ring = cycle(f_pop)
        mask = torch.zeros_like(f_pop, device=f_pop.device).type(torch.bool)
        for i, f in enumerate(f_pop):
            neighbors = torch.Tensor([next(ring), next(ring), next(ring)])
            elite_idx = self.problem.argopt(neighbors)
            mask[elite_idx + i] = True

        return mask

    def __star_select(self, f_pop):
        mask = torch.zeros_like(f_pop, device=f_pop.device).type(torch.bool)
        best = self.problem.argopt(f_pop, dim=0)
        mask[best] = True
        return mask

    

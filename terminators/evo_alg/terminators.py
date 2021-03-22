import torch

import terminators.base as base


class Convergence(base.Terminator):
    def __init__(self):
        super().__init__()

    def _criteria_met(self, agent):
        return torch.unique(agent.pop, dim=0).size(0) == 1

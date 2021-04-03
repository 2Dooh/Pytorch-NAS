import numpy as np

import terminators.base as base


class Convergence(base.Terminator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _criteria_met(self, agent):
        return self.force_termination or np.unique(agent.pop, axis=0).shape[0] == 1

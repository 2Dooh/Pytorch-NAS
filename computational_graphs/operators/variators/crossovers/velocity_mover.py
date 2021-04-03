from typing import Tuple

from utils.evo_alg.denormalization import denormalize

import torch

import computational_graphs.operators.base as base
import computational_graphs.operators.repairers.out_of_bounds_repair as repairers

class VelocityMover(base.OperatorBase):
    def __init__(self, 
                inertia_weight=0.7298,
                accelerate_const: Tuple[float, float] = (1.496618, 1.496618),
                max_velocity_rate=0.2,
                **kwargs):
        super().__init__(**kwargs)
        self.iw = inertia_weight
        self.ac = accelerate_const
        self.v_max = max_velocity_rate
        self.v = None
        self.init_v = False

    def __init_velocity(self, pop):
        xl, xu = self.problem.domain
        xl = -abs(xu - xl)
        xu = abs(xu - xl)
        v = torch.rand_like(pop)
        self.v = denormalize(v, xl, xu)


    def __call__(self,
                pop,
                offs, 
                elites, 
                **kwargs):
        if not self.init_v:
            self.init_v = True
            self.__init_velocity(pop)
        from random import random
        r_p, r_g = random(), random()
        c1, c2 = self.ac
        P = pop
        p = offs
        g = elites
        velocity = self.iw*self.v + c1*r_p * (p-P) + c2*r_g * (g-P)
        repair_out_of_bounds_manually = getattr(repairers, 'repair_out_of_bounds_manually')
    
        self.v = repair_out_of_bounds_manually(
            velocity, -self.v_max, self.v_max
        )

        pop += self.v
        repairer = getattr(repairers, 'OutOfBoundsRepair')
        pop = repairer(problem=self.problem, pop=pop, **kwargs)
        
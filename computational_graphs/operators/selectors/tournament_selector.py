import computational_graphs.operators.base as base



import numpy as np

class TournamentSelector(base.OperatorBase):
    def __init__(self, t_size=2, **kwargs):
        super().__init__(**kwargs)
        self.t_size = t_size
        # self.s_size = s_size
        # if self.s_size is not None \
        #     and self.s_size < self.t_size:
        #     self.logger.error(
        #         'Tournament Size must be smaller than Selection Size'
        #     )

    def __call__(self, f_pop, n_select, n_parents, **kwargs):
        optimum = self.problem.optimum
        indices = np.arange(f_pop.shape[0])
        self.s_size = self.s_size if self.s_size else f_pop.shape[0]
        selected = []

        while len(selected) < self.s_size:
            np.random.shuffle(indices)

            for start in range(0, f_pop.shape[0], self.t_size):
                tournament = indices[start:start+self.t_size]
                elite_idx = np.where(
                    (f_pop[tournament] == optimum(f_pop[tournament])).sum(axis=1) == f_pop.shape[1]
                )[0]
                selected += [np.random.choice(tournament[elite_idx])]

        return selected

        
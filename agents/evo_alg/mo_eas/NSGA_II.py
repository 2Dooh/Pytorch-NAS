import agents.evo_alg.ea_agent as base

import numpy as np


class NSGAII(base.EAAgent):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.rank_grp = None
        self.f_rank_grp = None
        self.f_pop = None

    def _initialize(self, **kwargs):
        super()._initialize(**kwargs)
        self.eval_dict['f_pop_obj'] = self.evaluate(pop=self.pop, **kwargs)
        self.rank_grp, self.f_rank_grp = self.__non_dominated_rank(self.pop, self.eval_dict['f_pop_obj'])

        rank, self.pop, self.eval_dict['f_pop_obj'] = self.__non_dominated_sort()
        CD = np.concatenate(list(map(self.__calc_crowding_distance, self.f_rank_grp.values())))
        self.f_pop = np.concatenate([rank[:, None], CD[:, None]], axis=1)
        
    def __non_dominated_sort(self):
        rank = np.concatenate([np.ones(self.rank_grp[i].shape[0]) * i for i in range(len(self.rank_grp))])
        pop = np.concatenate(list(self.rank_grp.values()))
        f_pop = np.concatenate(list(self.f_rank_grp.values()))
        return rank, pop, f_pop

    def non_dominated_rank(self):
        return self.__non_dominated_rank(
            self.pop, 
            self.eval_dict['f_pop_obj'])

    def __non_dominated_rank(self, pop, f_pop):
        rank_grp, f_rank_grp = {}, {}
        i = 0
        while pop.shape[0] != 0:
            count = self.__domination_count(f_pop)

            rank_grp[i] = pop[count == 0]
            f_rank_grp[i] = f_pop[count == 0]

            pop = pop[count != 0]
            f_pop = f_pop[count != 0]

            i += 1

        return rank_grp, f_rank_grp

    def __domination_count(self, f_pop):
        count = np.empty(f_pop.shape[0])
        for i in range(count.shape[0]):
            count[i] = self.problem._compare_vectorized(f_pop, f_pop[i]).sum()
            # y = sum([self.problem._compare_old(f_pop_j, f_pop[i]) for f_pop_j in f_pop])
            # assert(x == y)
        return count

    def _next(self, **kwargs):
        selection_mask = self.select(f_pop=self.f_pop, **kwargs)

        self.pop = self.pop[selection_mask]
        self.eval_dict['f_pop_obj'] = self.eval_dict['f_pop_obj'][selection_mask]

        self._step(pop=self.pop, offs=self.offs, **kwargs)
        
        self.eval_dict['f_offs_obj'] = self.evaluate(pop=self.offs, **kwargs)

        self.pop = np.concatenate((self.pop, self.offs))
        self.eval_dict['f_pop_obj'] = np.concatenate([self.eval_dict['f_pop_obj'], self.eval_dict['f_offs_obj']])

        self.rank_grp, self.f_rank_grp = self.__non_dominated_rank(self.pop, self.eval_dict['f_pop_obj'])
        self.rank_grp, self.f_rank_grp = self.__truncate_size()

        rank, self.pop, self.eval_dict['f_pop_obj'] = self.__non_dominated_sort()
        CD = np.concatenate(list(map(self.__calc_crowding_distance, self.f_rank_grp.values())))
        self.f_pop = np.concatenate([rank[:, None], CD[:, None]], axis=1)

        self._write_summary(**kwargs)

        super()._next(**kwargs)

    def _step(self, pop, offs, **kwargs):
        self.offs = self.mate(pop=pop, offs=offs, **kwargs)
        if self.mutator:
            self.offs = self.mutate(pop=self.offs, **kwargs)
        if self.repairer:
            self.offs = self.repair(pop=self.offs, problem=self.problem, **kwargs)
        

    def __truncate_size(self):
        n = 0
        rank_grp, f_rank_grp = {}, {}
        for i in range(len(self.rank_grp)):
            if n + self.rank_grp[i].shape[0] >= self.config.initializer.kwargs.size:
                n_takes = self.config.initializer.kwargs.size - n
                CD_i = self.__calc_crowding_distance(self.f_rank_grp[i])
                sorted_CD_i = CD_i.argsort()[::-1]
                rank_grp[i] = self.rank_grp[i][sorted_CD_i][:n_takes]
                f_rank_grp[i] = self.f_rank_grp[i][sorted_CD_i][:n_takes]
                break
            n += self.rank_grp[i].shape[0]
            rank_grp[i] = self.rank_grp[i]
            f_rank_grp[i] = self.f_rank_grp[i]

        return rank_grp, f_rank_grp

    @staticmethod
    def __calc_crowding_distance(F):
        CD = np.empty(F.shape[0])
        for f_i in range(F.shape[1]):
            f = F[:, f_i]
            Q = f.argsort()
            CD[Q[0]] = CD[Q[-1]] = np.inf
            for i in range(1, CD.shape[0]-1):
                CD[Q[i]] += f[Q[i + 1]] - f[Q[i - 1]]
        return CD

    def _write_summary(self, **kwargs):
        if self.summary_writer:
            for key, val in self.eval_dict.items():
                for obj in range(val.shape[1]):
                    self.summary_writer.add_scalars(
                        '{}/obj{}'.format(key, obj),
                        {
                            'max': val[:, obj].max(),
                            'min': val[:, obj].min(),
                            'mean': val[:, obj].mean(),
                            'std': val[:, obj].std()
                        },
                        self.current_gen
                    )
            self.summary_writer.close()

    def _save_checkpoint(self, checkpoint={}, **kwargs):
        checkpoint['rank_grp'] = self.rank_grp
        checkpoint['f_rank_grp'] = self.f_rank_grp
        super()._save_checkpoint(checkpoint=checkpoint, **kwargs)

    def _load_checkpoint(self, **kwargs):
        checkpoint = super()._load_checkpoint(**kwargs)
        self.rank_grp = checkpoint['rank_grp']
        self.f_rank_grp = checkpoint['f_rank_grp']
        
        return checkpoint


import agents.evo_alg.ea_agent as base

class PSO(base.EAAgent):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def _initialize(self, **kwargs):
        super()._initialize(**kwargs)
        self.offs = self.pop.clone()
        self.eval_dict = {}
        self.eval_dict['f_pop'] = self.evaluate(
            pop=self.pop
        ) 
        self.eval_dict['f_offs'] = self.eval_dict['f_pop'].clone()

        
    def _next(self, **kwargs):
        mask = self.problem._compare(
            Y1=self.eval_dict['f_pop'],
            Y2=self.eval_dict['f_offs']
        )
        if True in mask:
            self.offs[mask] = self.pop[mask]
            self.eval_dict['f_offs'][mask] = self.eval_dict['f_pop'][mask]

        elite_mask = self.select(
            f_pop=self.eval_dict['f_offs'],
            **kwargs
        )
        elites = self.offs[elite_mask]
        self._step(
            pop=self.pop,
            offs=self.offs,
            selection_mask=mask,
            elites=elites, 
            **kwargs
        )

        self._write_summary(**kwargs)
        self.eval_dict['f_pop'] = self.evaluate(
            pop=self.pop,
            **kwargs
        )
        super()._next(**kwargs)
        
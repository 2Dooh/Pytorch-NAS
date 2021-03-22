import computational_graphs.estimators.problems.base as base

class SingleObjectiveProblem(base.ProblemBase):
    def __init__(self, multi_dims=False, **kwargs):
        super().__init__(**kwargs)
        self.multi_dims = multi_dims

        self.opt = None
    
    @property
    def X_opt(self):
        pass
    @property
    def Y_opt(self):
        pass
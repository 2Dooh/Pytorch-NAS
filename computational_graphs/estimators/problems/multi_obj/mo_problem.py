import computational_graphs.estimators.problems.base as base

class MultiObjectiveProblem(base.ProblemBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def ranking_and_crowding_distance_compare(y1, y2):
        rank1, cd1 = y1
        rank2, cd2 = y2
        return (rank1 < rank2) or (rank1 == rank2 and cd1 > cd2)

    def optimum(self, Y):
        opt = Y[0]
        for y in Y:
            opt = opt if self.ranking_and_crowding_distance_compare(opt, y) else y
        return opt

    def arg_optimum(self, Y):
        argopt = 0
        for i, y_i in enumerate(Y):
            argopt = argopt if self.ranking_and_crowding_distance_compare(Y[argopt], y_i) else i
        return argopt

    def _calc_pareto_front(self, n_points):
        return None
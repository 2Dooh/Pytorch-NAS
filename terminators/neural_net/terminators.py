# import terminators.base as base

# class MaxEpoch(base.Terminator):
#     def __init__(self, n_epochs) -> None:
#         super().__init__()
#         self.n_epochs = n_epochs

#     def _criteria_met(self, agent):
#         return self.n_epochs == agent.current_epoch
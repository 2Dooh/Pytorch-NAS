import agents.neural_net.nn_agent as base

import torch

import os

class Regressor(base.NNAgent):
    def __init__(self, config):
        super().__init__(config)

    def _predict(self, outputs, **kwargs):
        pred, _ = outputs.max(1)
        return {'pred': pred}

    def _finalize(self, **kwargs):
        super()._finalize(**kwargs)

        torch.save(self.model.state_dict(), 
                   f=os.path.join(self.config.out_dir, 
                                  '{}-{}.pth.tar'.format(self.__class__.__name__, 
                                                         self.config.data_loader.name)))
        
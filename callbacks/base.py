import torch

import logging

class CallbackBase:
    def __init__(self) -> None:
        super().__init__()
        self.agent = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def _begin_fit(self, agent, **kwargs):
        self.agent = agent
    
    def _after_fit(self, **kwargs):
        if self.agent.config.exp_cfg.empty_cache:
            torch.cuda.empty_cache()

    def _begin_next(self, **kwargs):
        pass

    def _after_next(self, **kwargs):
        pass
from abc import abstractmethod, ABC

import logging

import torch
import torch.backends.cudnn as cudnn
from tensorboardX.writer import SummaryWriter

from callbacks import *

class Agent(ABC):
    def __init__(self, config, **kwargs):
        self.config = config
        self.logger = logging.getLogger(name=self.__class__.__name__)

        # set cuda flag
        has_cuda = torch.cuda.is_available()
        if has_cuda and not self.config.exp_cfg.cuda:
            self.logger.warning('CUDA device is available, but not utilized' )
        self.cuda = has_cuda and self.config.exp_cfg.cuda

        self.device = None
        self.summary_writer = None
        self.termination = None
        self.callback_handler = None


    ### Public Methods ###
    def solve(self, **kwargs):
        try:
            self._initialize(**kwargs)
            self.callback_handler.begin_fit(agent=self, **kwargs)
            while not self.termination._criteria_met(self, **kwargs):
                self._next(**kwargs)
            self._finalize(**kwargs)
        except KeyboardInterrupt:
            self.logger.info('You have entered CTRL+C... Wait to finalize')
            self._finalize(**kwargs)
    ### Public Methods ###

    ### Virtual Methods ###
    def _initialize(self, **kwargs):
        torch.manual_seed(self.config.exp_cfg.seed)
        
        if self.cuda:
            cudnn.enabled = True
            cudnn.benchmark = not self.config.exp_cfg.deterministic
            cudnn.deterministic = self.config.exp_cfg.deterministic
            if self.config.exp_cfg.deterministic:
                self.logger.info('Applying deterministic mode; cudnn disabled!')

        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.logger.info('Program will run on *****{}*****'.format(self.device))

        if self.config.exp_cfg.summary_writer:
            self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir)
            self._setup_summary_writer(**kwargs)

    def _load_checkpoint(self, path, **kwargs):
        checkpoint = torch.load(path, self.device, **kwargs)
        return checkpoint

    def _finalize(self, **kwargs):
        self.logger.info('Please wait while finalizing...')
        # self._save_checkpoint(
        #     self.model.state_dict(),
        #     os.path.join(self.config.out_dir, 'model.pth')
        # )
        self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        self.summary_writer.close()
    
    def _save_checkpoint(self, checkpoint, filepath, **kwargs):
        torch.save(checkpoint, os.path.join(self.config.checkpoint_dir, filepath), **kwargs)

    def _setup_summary_writer(self, **kwargs):
        pass
    ### Virtual Methods ###
    

    ### Abstract Methods ###
    @abstractmethod
    def _next(self, **kwargs):
        raise NotImplementedError
    @abstractmethod
    def _write_summary(self, **kwargs):
        raise NotImplementedError
    ### Abstract Methods ###
    
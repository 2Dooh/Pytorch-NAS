from abc import abstractmethod, ABC

import logging

from tensorboardX.writer import SummaryWriter



class AgentBase(ABC):
    def __init__(self, config, **kwargs):
        self.config = config
        self.logger = logging.getLogger(name=self.__class__.__name__)

        self.summary_writer = None
        self.callback_handler = None


    ### Public Methods ###
    def solve(self, **kwargs):
        try:
            self._initialize(**kwargs)
            self.callback_handler.begin_fit(agent=self, **kwargs)
            while not self.termination._criteria_met(self, **kwargs):
                self.callback_handler.begin_next(**kwargs)
                self._next(**kwargs)
                self.callback_handler.after_next(**kwargs)
            self._finalize(**kwargs)
        except KeyboardInterrupt:
            self.logger.info('You have entered CTRL+C... Wait to finalize')
            self._finalize(**kwargs)
        except Exception as e:
            self.logger.error(e, exc_info=True)
            self._finalize(**kwargs)
    ### Public Methods ###

    ### Virtual Methods ###
    def _initialize(self, **kwargs):
        if self.config.exp_cfg.summary_writer:
            self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir)
            self._setup_summary_writer(**kwargs)

    def _load_checkpoint(self, api, path, **kwargs):
        checkpoint = api.load(path, **kwargs)
        return checkpoint

    def _finalize(self, **kwargs):
        self.logger.info('Please wait while finalizing...')
        # self._save_checkpoint(
        #     self.model.state_dict(),
        #     os.path.join(self.config.out_dir, 'model.pth')
        # )
        # self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
        # self.summary_writer.close()
    
    def _save_checkpoint(self, api, **kwargs):
        api.save(**kwargs)

    def _setup_summary_writer(self, **kwargs):
        pass
    def _next(self, **kwargs):
        pass
    ### Virtual Methods ###
    

    ### Abstract Methods ###
    
    @abstractmethod
    def _write_summary(self, **kwargs):
        raise NotImplementedError
    ### Abstract Methods ###
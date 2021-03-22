from typing import Dict, final

import agents.base as base

import computational_graphs.models as graph_models

import data_loaders

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler

from abc import abstractmethod

from enum import Enum

import callbacks.neural_net as callbacks

import os


class Mode(Enum):
    TRAIN = True
    EVAL = False


class NNAgent(base.AgentBase):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        # set cuda flag
        has_cuda = torch.cuda.is_available()
        if has_cuda and not self.config.exp_cfg.cuda:
            self.logger.warning('CUDA device is available, but not utilized' )
        self.cuda = has_cuda and self.config.exp_cfg.cuda

        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        
        self.current_epoch = self.n_steps = 0

        self.criterion = getattr(
            nn, self.config.criterion.name, None
        )(**self.config.criterion.kwargs)

        self.callback_handler = getattr(
            callbacks,
            'NNCallbackHandler',
        )([getattr(callbacks, cb)(**kw) for cb, kw in self.config.callbacks.items()])

        data_loader = getattr(
            data_loaders, self.config.data_loader.name
        )(**self.config.data_loader.kwargs)
        self.train_queue = data_loader.train_loader
        self.valid_queue = data_loader.test_loader
 
        self.model = None
        self.optimizer = None
        self.scheduler = None

    ### Public Methods ###
    def loop(self, train=True, **kwargs):
        mode, queue, context_manager, propagate_func = \
            self.__config_run_mode(train, **kwargs)
        self.callback_handler.begin_loop(mode=mode, queue=queue, **kwargs)
        with context_manager():
            for step, (inputs, targets) in enumerate(queue):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                results = propagate_func(
                    step=step, 
                    inputs=inputs, 
                    targets=targets, 
                    **kwargs
                )

                predicted = self._predict(**results, **kwargs)
                self.callback_handler.after_batch(
                    step=step, 
                    real=targets,
                    **predicted,
                    **results,
                    **kwargs
                )
        self.callback_handler.after_loop(**kwargs)
        self._write_summary(mode=mode, **kwargs)
    ### Public Methods ###

    ### Virtual Methods ###
    def _initialize(self, **kwargs):
        super()._initialize(**kwargs)
        torch.manual_seed(self.config.exp_cfg.seed)
        
        if self.cuda:
            cudnn.enabled = True
            cudnn.benchmark = not self.config.exp_cfg.deterministic
            cudnn.deterministic = self.config.exp_cfg.deterministic
            if self.config.exp_cfg.deterministic:
                self.logger.info('Applying deterministic mode; cudnn disabled!')

        self.logger.info('Program will run on *****{}*****'.format(self.device))
    
        self.model = \
            getattr(graph_models, self.config.model.name)(**self.config.model.kwargs)
        self.optimizer = getattr(
            torch.optim, self.config.optimizer.name, None
        )(self.model.parameters(), **self.config.optimizer.kwargs)

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)

        if 'checkpoint' in self.config:
            self._load_checkpoint(self.config.checkpoint, **kwargs)
        if 'scheduler' in self.config:
            self.scheduler = getattr(
                lr_scheduler, self.config.scheduler.name
            )(self.optimizer, **self.config.scheduler.kwargs)

    def _load_checkpoint(self, **kwargs):
        checkpoint = super()._load_checkpoint(api=torch, device=self.device, **kwargs)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.criterion = checkpoint['criterion']
        if 'eval_dict' in checkpoint:
            self.agent.eval_dict = checkpoint['eval_dict']
            
        return checkpoint

    def _save_checkpoint(self, scores, checkpoint={}, **kwargs):
        checkpoint['epoch'] = self.current_epoch
        checkpoint['model_state_dict']= self.model.state_dict()
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        checkpoint['criterion'] = self.criterion
        checkpoint['eval_dict'] = self.eval_dict

        filepath = '[{}] Ep_{}-Scores-{}.pth.tar'.format(
            self.__class__.__name__, self.current_epoch, scores
        )
        super()._save_checkpoint(
            api=torch,
            obj=checkpoint, 
            f=os.path.join(self.config.checkpoint_dir, filepath), 
            **kwargs)

    def _model_train(self, **kwargs) -> Dict:
        outputs, loss = self._model_infer(**kwargs).values()
        self.back_prop(loss, **kwargs)
        self.step(self.optimizer, **kwargs)

        return {'outputs': outputs, 'loss': loss}

    def _model_infer(self, **kwargs) -> Dict:
        outputs, loss = self.feed_forward(model=self.model, criterion=self.criterion, **kwargs)
        return {'outputs': outputs, 'loss': loss}

    def _write_summary(self, mode, **kwargs):
        if self.summary_writer:
            for key, val in enumerate(self.eval_dict[mode.name].items()):
                self.summary_writer.add_scalar(
                    '{}/{}'.format(key, mode.name),
                    val,
                    self.current_epoch
                )
            self.summary_writer.close()
            # for key, val in self.eval_dict[mode.name].items():
            #     self.summary_writer.add_scalars(
            #         '{}/{}'.format(key, mode.name),
            #         val,
            #         self.current_epoch
            #     )

    def _finalize(self, **kwargs):
        super()._finalize(**kwargs)
        self.logger.info(self.eval_dict)
    ### Virtual Methods ###

    ### Abstract Methods
    @abstractmethod
    def _predict(self, **kwargs) -> Dict:
        raise NotImplementedError
    ### Abstract Methods

    ### Private Methods
    @final
    def feed_forward(self,
                    model, 
                    criterion, 
                    inputs, 
                    targets,
                    **kwargs):
        self.callback_handler.begin_forward(
            model=model, 
            criterion=criterion,
            inputs=inputs,
            targets=targets,
            **kwargs
        )
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        return outputs, loss

    @final
    def _next(self, **kwargs):
        self.loop(**kwargs)
        if self.current_epoch % self.config.exp_cfg.eval_freq == 0:
            self.loop(train=False, **kwargs)
        self.current_epoch += 1

    @final
    def back_prop(self, loss, **kwargs):
        loss.backward()
        self.callback_handler.after_backward(loss=loss, **kwargs)

    @final
    def step(self, optimizer, **kwargs):
        self.callback_handler.begin_step(optimizer=optimizer, **kwargs)

        optimizer.step()
        optimizer.zero_grad()

        self.n_steps += 1
        self.callback_handler.after_step(optimizer=optimizer, **kwargs)

    @final
    def __config_run_mode(self, train=True, **kwargs):
        if not train:
            mode = Mode.EVAL
            self.model.eval()
            queue = self.valid_queue
            context_mananger = torch.no_grad
            propagate_func = self._model_infer
            return mode, queue, context_mananger, propagate_func
        
        mode = Mode.TRAIN
        self.model.train()
        queue = self.train_queue
        context_mananger = torch.enable_grad
        propagate_func = self._model_train

        return mode,queue,context_mananger,propagate_func
    ### Private Methods

    

        

from typing import Dict

import utils.neural_net.metrics as metric_lib
from utils.neural_net.flops_benchmark import add_flops_counting_methods

import callbacks.base as base

from pprint import pformat

import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_

import time

class NeuralNetCallback(base.CallbackBase):
    def __init__(self) -> None:
        super().__init__()
        self.agent = None
    
    def _begin_step(self, **kwargs):
        pass
    
    def _after_step(self, **kwargs):
        pass

    def _after_backward(self, **kwargs):
        pass

    def _after_forward(self, **kwargs):
        pass

    def _begin_forward(self, **kwargs):
        pass

    def _after_batch(self, step, **kwargs):
        pass

    def _after_fit(self, **kwargs):
        pass

    def _after_loop(self, **kwargs):
        pass

    def _begin_loop(self, **kwargs):
        pass

class NNCallbackHandler:
    def __init__(self, callbacks=None) -> None:
        super().__init__()
        self.callbacks = callbacks if callbacks else []

    def begin_fit(self, agent, **kwargs):
        for callback in self.callbacks:
            callback._begin_fit(agent=agent, **kwargs)

    def after_fit(self, **kwargs):
        for callback in self.callbacks:
            callback._after_fit(**kwargs)

    def begin_next(self, **kwargs):
        for callback in self.callbacks:
            callback._begin_next(**kwargs)
    
    def begin_loop(self, **kwargs):
        for callback in self.callbacks:
            callback._begin_loop(**kwargs)

    def after_loop(self, **kwargs):
        for callback in self.callbacks:
            callback._after_loop(**kwargs)

    def after_next(self, **kwargs):
        for callback in self.callbacks:
            callback._after_next(**kwargs)

    def begin_step(self, **kwargs):
        for callback in self.callbacks:
            callback._begin_step(**kwargs)

    def after_step(self, **kwargs):
        for callback in self.callbacks:
            callback._after_step(**kwargs)

    def after_backward(self, **kwargs):
        for callback in self.callbacks:
            callback._after_backward(**kwargs)

    def begin_forward(self, **kwargs):
        for callback in self.callbacks:
            callback._begin_forward(**kwargs)

    def after_batch(self, **kwargs):
        for callback in self.callbacks:
            callback._after_batch(**kwargs)

class LearningRateScheduler(NeuralNetCallback):
    def __init__(self) -> None:
        super().__init__()

    def _after_next(self, **kwargs):
        self.agent.scheduler.step()
        self.logger.info(
            'Ep: {} - lr: {}'.format(
                self.agent.current_epoch+1,
                self.agent.scheduler.get_last_lr()[0]
            )
        )

class GradientClipper(NeuralNetCallback):
    def __init__(self, clip_val) -> None:
        super().__init__()
        self.clip_val = clip_val

    def _after_backward(self, **kwargs):
        clip_grad_norm_(
            self.agent.model.parameters(),
            self.clip_val
        )

class ModelComplexity(NeuralNetCallback):
    def __init__(self) -> None:
        super().__init__()
        self.n_params = self.n_flops = 0
        self.start_time = self.infer_time = 0
        self.avg_infer_time = metric_lib.AverageMeter()
        self.mode = None

    def _begin_fit(self, agent, **kwargs):
        super()._begin_fit(agent, **kwargs)
        if not self.agent.eval_dict:
            self.agent.eval_dict = {}
        self.n_params = \
            np.sum(
                np.prod(v.size()) \
                    for v in filter(
                        lambda p: p.requires_grad, 
                        self.agent.model.parameters()
                    )
            ) / 1e6

        self.agent.model = \
            add_flops_counting_methods(self.agent.model)
        self.agent.model.eval()
        self.agent.model.start_flops_count()

        data_iter = iter(self.agent.valid_queue)
        x, _ = data_iter.next()
        random_data = torch.rand(1, *x[0].size())
        self.agent.model(random_data.to(self.agent.device))
        self.n_flops = np.round(self.agent.model.compute_average_flops_cost() / 1e6, 4)
        
        self.agent.eval_dict.update({'n_params': self.n_params, 'n_flops': self.n_flops})

        self.logger.info(pformat(self.agent.eval_dict))
    
    def _begin_loop(self, mode, **kwargs):
        self.mode = mode
    
    def _begin_forward(self, **kwargs):
        if self.is_evaluating():
            self.start_time = time.time()

    def is_evaluating(self):
        return not self.mode.value

    def _after_forward(self, **kwargs):
        end = time.time()
        if self.is_evaluating():
            self.infer_time += end - self.start_time

    def _after_loop(self, **kwargs):
        if self.is_evaluating():
            self.avg_infer_time._update(self.infer_time)

    def _after_fit(self, **kwargs):
        self.agent.eval_dict['avg_infer_time'] = \
            self.avg_infer_time.get_value
        
    
import numpy as np
class ModelSaver(NeuralNetCallback):
    def __init__(self, metrics: Dict = {'ErrorRate': {}}, thresholds=None) -> None:
        super().__init__()
        self.thresholds = [10] * len(metrics) if not thresholds else thresholds
        self.metrics = \
            [getattr(metric_lib, metric)(**kwargs) for metric, kwargs in metrics.items()]
        self.evaluated = False
        self.mode = None


    def _begin_loop(self, mode, **kwargs):
        self.evaluated = not mode.value
        self.mode = mode

    @property
    def set_threshold(self, thresholds):
        self.thresholds = thresholds

    def _begin_next(self, **kwargs):
        if self.evaluated:
            thresholds = np.array(self.thresholds)
            new_scores = np.array(list(self.agent.eval_dict[self.mode.name].values()))
            if self.dominate(new_scores, thresholds):
                dominated_indices = np.where(new_scores < thresholds)[0]
                list_scores = list(self.agent.eval_dict[self.mode.name].items())
                scores_str = '-'.join('{}_{:.3f}'.format(*list_scores[idx]) for idx in dominated_indices)
                self.thresholds = new_scores.tolist()
                self.agent._save_checkpoint(scores=scores_str)


    @staticmethod
    def dominate(score1, score2):
        not_dominated = score1 <= score2
        dominate = score1 < score2
        return not_dominated.all() and True in dominate

    
    


class ModelEvaluator(NeuralNetCallback):
    def __init__(self, metrics: Dict = {'ErrorRate': {}}) -> None:
        super().__init__()
        self.metrics = \
            [getattr(metric_lib, metric)(**kwargs) for metric, kwargs in metrics.items()]
        self.message = '({}) Ep {}:'
        self.n_batches = None
        self.dataset_size = None
        self.current_iterations = 0
        self.mode = None

    def _begin_fit(self, agent, **kwargs):
        super()._begin_fit(agent, **kwargs)
        if not self.agent.eval_dict:
            self.agent.eval_dict = {}

    def _begin_loop(self, mode, queue, **kwargs):
        self.mode = mode
        self.n_batches = len(queue)
        self.dataset_size = len(queue.dataset)
        self.current_iterations = 0
        for metric in self.metrics:
            metric.reset()

    def _after_batch(self, step, pred, loss, **kwargs):
        self.current_iterations += pred.size(0)
        for metric in self.metrics:
            metric._update(step=step, pred=pred, loss=loss.item(), **kwargs)
        

        print_flag = self.n_batches // self.agent.config.exp_cfg.log_interval
        print_flag = max(1, print_flag)

        if (step+1) % print_flag == 0:
            message = \
                (self.message + '[(I){}/{} - (B){}/{} ({:.0f}%)]\tLoss: {:.6f}').format(
                    self.mode.name,
                    self.agent.current_epoch+1,

                    self.current_iterations,
                    self.dataset_size,

                    step+1,
                    self.n_batches,

                    100. * (step+1) / self.n_batches,
                    loss.item()
                )
            self.logger.info(message)
        
    def _after_loop(self, **kwargs):
        if self.mode.name not in self.agent.eval_dict:
            self.agent.eval_dict[self.mode.name] = {}
        message = self.message.format(self.mode.name, self.agent.current_epoch+1)
        for metric in self.metrics:
            name = metric.__class__.__name__
            # self.agent.eval_dict[name][self.mode.name] = metric.get_value
            self.agent.eval_dict[self.mode.name][name] = metric.get_value
            message += ' Avg {}: {:.3f} -'.format(
                name, self.agent.eval_dict[self.mode.name][name]
            )

        self.logger.info(message[:-1] + '\n')
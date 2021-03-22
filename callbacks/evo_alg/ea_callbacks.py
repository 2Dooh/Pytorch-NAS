import callbacks.base as base

from pprint import pformat

import time

class EvoAlgCallback(base.CallbackBase):
    def __init__(self) -> None:
        super().__init__()
        self.agent = None

    def _begin_eval(self, **kwargs):
        pass
    
    def _after_eval(self, **kwargs):
        pass

    def _begin_select(self, **kwargs):
        pass

    def _after_select(self, **kwargs):
        pass

    def _begin_crossover(self, **kwargs):
        pass

    def _after_crossover(self, **kwargs):
        pass

    def _begin_survival_select(self, **kwargs):
        pass

    def _after_survival_select(self, **kwargs):
        pass

    def _begin_mutate(self, **kwargs):
        pass

    def _after_mutate(self, **kwargs):
        pass

class EACallbackHandler:
    def __init__(self, callbacks=None) -> None:
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

    def after_next(self, **kwargs):
        for callback in self.callbacks:
            callback._after_next(**kwargs)

    def begin_eval(self, **kwargs):
        for callback in self.callbacks:
            callback._begin_eval(**kwargs)
    
    def after_eval(self, **kwargs):
        for callback in self.callbacks:
            callback._after_eval(**kwargs)

    def begin_select(self, **kwargs):
        for callback in self.callbacks:
            callback._begin_select(**kwargs)

    def after_select(self, **kwargs):
        for callback in self.callbacks:
            callback._after_select(**kwargs)

    def begin_crossover(self, **kwargs):
        for callback in self.callbacks:
            callback._begin_crossover(**kwargs)

    def after_crossover(self, **kwargs):
        for callback in self.callbacks:
            callback._after_crossover(**kwargs)

    def begin_survival_select(self, **kwargs):
        for callback in self.callbacks:
            callback._begin_survival_select(**kwargs)

    def after_survival_select(self, **kwargs):
        for callback in self.callbacks:
            callback._after_survival_select(**kwargs)

    def begin_mutate(self, **kwargs):
        for callback in self.callbacks:
            callback._begin_mutate(**kwargs)

    def after_mutate(self, **kwargs):
        for callback in self.callbacks:
            callback._after_mutate(**kwargs)

import matplotlib.pyplot as plt

import numpy as np

from itertools import combinations

import os

class ParetoFrontProgress(EvoAlgCallback):
    def __init__(self, 
                plot_freq=1, 
                n_points=100,
                labels=None,
                loc='upper right',
                **kwargs):
        super().__init__(**kwargs)
        self.plot_freq = plot_freq
        self.loc = loc
        self.n_points = n_points
        self.labels = labels
        self.plot_info = None
        self.has_front = False

    def _begin_fit(self,  agent, **kwargs):
        super()._begin_fit(agent=agent, **kwargs)
        os.makedirs(os.path.join(self.agent.config.checkpoint_dir, 'gifs'), exist_ok=True)  
        if not self.labels:
            n_obj = self.agent.problem.n_obj
            self.labels = \
                [
                    r'$f_{}(x)$'.format(
                        (i+1)) for i in range(n_obj)
                ]
        points = \
            self.agent.problem._calc_pareto_front(n_points=self.n_points)

        points = list(combinations(points, r=2))
        
        n_obj = list(combinations(range(n_obj), r=2))
        ax_labels = list(combinations(self.labels, r=2))
        if len(points) == 0:
            points = [None] * len(ax_labels)
        fig_ax = []
        for i in range(len(ax_labels)):
            fig, ax = plt.subplots(num=i)
            fig_ax += [[fig, ax]]
        self.plot_info = [n_obj, ax_labels, fig_ax, points]

            

    def _begin_next(self, **kwargs):
        if self.agent.current_gen % self.plot_freq == 0:
            f_pop = self.agent.eval_dict['f_pop_obj']
            for i, (obj_pair, labels, fig_ax, data) in enumerate(zip(*self.plot_info)):
                fig, ax = fig_ax
                ax.set_xlabel(labels[0])
                ax.set_ylabel(labels[1])
                if data:
                    ax.plot(*data, label='pareto front', color='red')
                ax.grid(True, linestyle='--')
                ax.legend(loc=self.loc)
                X = f_pop[:, obj_pair[0]]
                Y = f_pop[:, obj_pair[1]]
                # lim = ax.get_xlim(), ax.get_ylim()
                ax.plot(X, Y, 'g.', label='gen: {}'.format(self.agent.current_gen))
                fig.savefig(
                    os.path.join(
                        self.agent.config.checkpoint_dir,
                        'gifs',
                        '[{}][{}]_Gen_{:0>3d}.pdf'.format(
                            self.agent.__class__.__name__,
                            self.agent.problem.__class__.__name__,
                            self.agent.current_gen
                        )
                    )
                )
                ax.clear()

            
class PopLogger(EvoAlgCallback):
    def __init__(self, log_freq=1, **kwargs):
        super().__init__(**kwargs)
        self.log_freq = log_freq
        self.start = None

    def _begin_next(self, **kwargs):
        self.start = time.time()

    def _after_next(self, **kwargs):
        if self.agent.current_gen % self.log_freq == 0:
            end = time.time() - self.start
            info = {
                'current_gen': self.agent.current_gen,
                'n_evals': self.agent.n_evals,
                'time': end
            }
            self.logger.info(pformat(info))

class PopSaver(EvoAlgCallback):
    def __init__(self, save_freq=1, **kwargs):
        super().__init__(**kwargs)
        self.save_freq = save_freq

    def _after_next(self, **kwargs):
        if self.agent.current_gen % self.save_freq == 0:
            self.agent._save_checkpoint()

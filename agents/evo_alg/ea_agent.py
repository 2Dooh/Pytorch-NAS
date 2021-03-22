from typing import final
import agents.base as base

import callbacks.evo_alg as callbacks

import computational_graphs.operators.initializers as initializers
import computational_graphs.operators.optimizers as optimizers
import computational_graphs.operators.selectors as selectors
import computational_graphs.estimators.problems as problems


import numpy as np
import random

import torch

import os

class EAAgent(base.AgentBase):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.current_gen = self.n_evals = 0

        self.problem = getattr(
            problems, self.config.problem.name
        )(**self.config.problem.kwargs)

        self.initializer = getattr(
            initializers, self.config.initializer.name
        )(**self.config.initializer.kwargs, problem=self.problem)
        self.mutator = None
        if 'mutator' in self.config:
            self.mutator = getattr(
                optimizers, 
                self.config.mutator.name
            )(**self.config.mutator.kwargs, problem=self.problem)

        if 'callbacks' in self.config:
            self.callback_handler = getattr(
                callbacks,
                'EACallbackHandler',
            )([getattr(callbacks, cb)(**kw) for cb, kw in self.config.callbacks.items()])
        

        self.selector = getattr(
            selectors, self.config.selector.name
        )(**self.config.selector.kwargs, problem=self.problem)
        self.crossover = getattr(
            optimizers, self.config.crossover.name
        )(**self.config.crossover.kwargs, problem=self.problem)

        self.population = None
        self.offsprings = None

    def _write_summary(self, **kwargs):
        if self.summary_writer:
            for key, val in self.eval_dict.items():
                self.summary_writer.add_scalars(
                    main_tag=key,
                    tag_scalar_dict={
                        'mean': val.mean().item(),
                        'std': val.std().item(),
                        'max': val.max().item(),
                        'min': val.min().item()
                    },
                    global_step=self.current_gen
                )
        self.summary_writer.close()

    
    def _initialize(self, **kwargs):
        super()._initialize(**kwargs)
        np.random.seed(self.config.exp_cfg.seed)
        random.seed(self.config.exp_cfg.seed)

        self.population = self.initializer()
        # self.population = self.population.to(self.device)
        if 'checkpoint' in self.config:
            self._load_checkpoint(path=self.config.checkpoint, **kwargs)
    

    def _next(self, **kwargs):
        self.current_gen += 1

    def _finalize(self, **kwargs):
        super()._finalize(**kwargs)
        self.logger.info(self.eval_dict)

    @final
    def evaluate(self, population, **kwargs):
        self.callback_handler.begin_eval(**kwargs)
        f = self.problem.eval(pop=population, **kwargs)
        self.callback_handler.after_eval(**kwargs)
        self.n_evals += f.shape[0]
        return f

    @final
    def select(self, f_pop, **kwargs):
        self.callback_handler.begin_select(**kwargs)
        boolean_mask = self.selector(
            f_pop=f_pop, **kwargs
        )
        self.callback_handler.after_select(
            f_pop=f_pop,
            boolean_mask=boolean_mask,
            **kwargs
        )
        return boolean_mask

    @final
    def mate(self, **kwargs):
        self.callback_handler.begin_crossover(**kwargs)
        res = self.crossover(**kwargs)
        self.callback_handler.after_crossover(res=res, **kwargs)
        return res

    @final
    def mutate(self, **kwargs):
        self.callback_handler.begin_mutate(**kwargs)
        res = self.mutator(**kwargs)
        self.callback_handler.after_mutate(res=res, **kwargs)
        return res

    @final
    def survival_select(self, **kwargs):
        self.callback_handler.begin_survival_select(**kwargs)
        self.survival_selector(**kwargs)
        self.callback_handler.after_survival_select(**kwargs)

    def _step(self, **kwargs):
        self.mate(**kwargs)
        if self.mutator: 
            self.mutate(**kwargs)
        
    def _save_checkpoint(self, checkpoint={}, **kwargs):
        checkpoint['gen'] = self.current_gen
        checkpoint['pop'] = self.population
        checkpoint['offs'] = self.offsprings
        checkpoint['eval_dict'] = self.eval_dict

        filepath = '[{}] Gen_{}.pth.tar'.format(
            self.__class__.__name__,
            self.current_gen
        )
        super()._save_checkpoint(api=torch, obj=checkpoint, f=os.path.join(self.config.checkpoint_dir, filepath))

    def _load_checkpoint(self, **kwargs):
        checkpoint = super()._load_checkpoint(api=torch, **kwargs)
        self.population = checkpoint['pop']
        self.offsprings = checkpoint['offs']
        self.current_gen = checkpoint['gen']
        self.eval_dict = checkpoint['eval_dict']

        return checkpoint



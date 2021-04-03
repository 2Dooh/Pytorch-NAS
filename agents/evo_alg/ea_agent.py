from typing import final
import agents.base as base

import callbacks.evo_alg as callbacks

import computational_graphs.operators.initializers as initializers
import computational_graphs.operators.variators as variators
import computational_graphs.operators.selectors as selectors
import computational_graphs.operators.repairers as repairers
import computational_graphs.estimators.problems as problems

from computational_graphs.operators.base import OperatorBase

import numpy as np
import random

import torch

import os

from pprint import pformat

class EAAgent(base.AgentBase):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.problem = getattr(problems, self.config.problem.name)(**self.config.problem.kwargs)

        self.mutator = \
            getattr(variators, 
                    self.config.mutator.name, 
                    OperatorBase(None))(**self.config.mutator.kwargs, 
                                        problem=self.problem)
        self.repairer = \
            getattr(repairers, 
                    self.config.repairer.name, 
                    OperatorBase(None))(**self.config.repairer.kwargs, 
                                        problem=self.problem)

        self.initializer = \
            getattr(initializers, 
                    self.config.initializer.name)(**self.config.initializer.kwargs, 
                                                  repairer=self.repairer, 
                                                  problem=self.problem)

        self.selector = \
            getattr(selectors, 
                    self.config.selector.name)(**self.config.selector.kwargs, 
                                               problem=self.problem)
        self.crossover = \
            getattr(variators, 
                    self.config.crossover.name)(**self.config.crossover.kwargs, 
                                                problem=self.problem)

        if 'callbacks' in self.config:
            self.callback_handler = getattr(
                callbacks,
                'EACallbackHandler',
            )([getattr(callbacks, cb)(**kw) for cb, kw in self.config.callbacks.items()])
        
        if 'n_offs' in self.config.ea_cfg:
            self.n_offs = self.config.ea_cfg.n_offs
        else:
            self.n_offs = self.config.initializer.kwargs.size
        
        if 'min_infeas_pop_size' in self.config.ea_cfg:
            self.min_infeas_pop_size = self.config.ea_cfg.min_infeas_pop_size
        else:
            self.min_infeas_pop_size = 0

        self.pop = self.offs = None
        self.current_gen = self.n_evals = 0

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

        self.pop = self.initializer()
        # self.pop = self.repair(pop=self.pop)
        # if self.repairer:
        
        # self.pop = self.pop.to(self.device)
        if 'checkpoint' in self.config:
            self._load_checkpoint(path=self.config.checkpoint, **kwargs)
    

    def _next(self, **kwargs):
        self.offs = self._step(pop=self.pop, **kwargs)

        if self.offs.shape[0] == 0:
            self.terminator.force_termination = True
            return
        elif self.offs.shape[0] < self.n_offs:
            self.logger.warn('Could not produce the required number of unique offs!')

        self.eval_dict['F_offs'] = self.evaluate(pop=self.offs, **kwargs)

        self.pop = np.concatenate([self.pop, self.offs])

        if self.survival_selector:
            self.pop = self.survival_select(pop=self.pop, n_min_infeas_survive=self.n_min_infeas_pop_size)
        
        self.current_gen += 1

    def _finalize(self, **kwargs):
        super()._finalize(**kwargs)
        self.logger.info(pformat(self.eval_dict))
        # self.logger.info(pformat(self.problem.indices_dict))
        # self.logger.info('Architecture evaluated: {}'.format(len(self.problem.indices_dict)))

    @final
    def evaluate(self, pop, **kwargs):
        self.callback_handler.begin_eval(**kwargs)
        f = self.problem.eval(pop=pop, **kwargs)
        self.callback_handler.after_eval(**kwargs)
        self.n_evals += f.shape[0]
        return f

    @final
    def variate(self, pop, f_pop, n_remaining, parents=None, **kwargs):
        if not parents:
            n_select = n_remaining // self.n_offs
            parents = self.select(f_pop, n_select, **kwargs)
        pass

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

    def _step(self, pop, **kwargs):
        offs = []
        n_infills = 0
        while len(offs) < self.n_offs:
            n_remaining = self.n_offs - len(offs)
            _off = self.variate(n_remaining, **kwargs)
            _off = self.repair(pop=_off, **kwargs)
            offs = np.concatenate([offs, _off])
            offs = np.unique(offs, axis=0)

            if offs.shape[0] > self.config.initializer.kwargs.size:
                n_remaining = self.config.initializer.kwargs.size - offs.shape[0]
                _off = _off[:n_remaining]
            

            n_infills += 1
            if n_infills > 100:
                break

        return offs
        # self.mate(**kwargs)
        # if self.mutator: 
        #     self.mutate(**kwargs)
        
    def _save_checkpoint(self, checkpoint={}, **kwargs):
        checkpoint['gen'] = self.current_gen
        checkpoint['pop'] = self.pop
        checkpoint['offs'] = self.offs
        checkpoint['eval_dict'] = self.eval_dict

        filepath = '[{}] Gen_{}.pth.tar'.format(
            self.__class__.__name__,
            self.current_gen
        )
        super()._save_checkpoint(api=torch, obj=checkpoint, f=os.path.join(self.config.checkpoint_dir, filepath))

    def _load_checkpoint(self, **kwargs):
        checkpoint = super()._load_checkpoint(api=torch, **kwargs)
        self.pop = checkpoint['pop']
        self.offs = checkpoint['offs']
        self.current_gen = checkpoint['gen']
        self.eval_dict = checkpoint['eval_dict']

        return checkpoint



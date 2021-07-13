import pymoo.factory as factory

import procedures.operators.repair as repairers

import logging

import copy

from pymoo.model.repair import NoRepair
from pymoo.model.duplicate import NoDuplicateElimination, DefaultDuplicateElimination

import procedures.problems as custom_problems
import procedures.operators.dupplicate as duplicate_eliminators

import agents.base as base

import torch

import callbacks.evo_alg as callbacks

import os

from utils.prepare_seed import prepare_seed

class EvoAgent(base.AgentBase):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.sampling = factory.get_sampling(config.sampling)
        self.crossover = factory.get_crossover(name=config.crossover.name, **config.crossover.kwargs)
        self.mutation = factory.get_mutation(name=config.mutation.name, **config.mutation.kwargs)

        self.repair = NoRepair()
        if 'repair' in config:
            self.repair = getattr(repairers, config.repair)()
        self.eliminate_duplicates = NoDuplicateElimination()
        if 'eliminate_duplicates' in config:
            if isinstance(config.eliminate_duplicates, bool):
                if config.eliminate_duplicates:
                    self.eliminate_duplicates = \
                        DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = \
                    getattr(duplicate_eliminators, 
                            config.eliminate_duplicates.name)(**config.eliminate_duplicates.kwargs)
        # algorithm = get_algorithm(config.algorithm.name)
        self.algorithm = factory.get_algorithm(name=config.algorithm.name,
                                               sampling=self.sampling,
                                               crossover=self.crossover,
                                               mutation=self.mutation,
                                               repair=self.repair,
                                               eliminate_duplicates=self.eliminate_duplicates,
                                               **config.algorithm.kwargs)

        self.termination = factory.get_termination(config.termination.name, *config.termination.args)

        if config.exp_cfg.custom_problem:
            self.problem = getattr(custom_problems, config.problem.name)(**config.problem.kwargs)
        else:
            self.problem = factory.get_problem(name=config.problem.name, **config.problem.kwargs)

        self.obj = None
        if 'callbacks' in self.config:
            self.callback_handler = getattr(
                callbacks,
                'EACallbackHandler',
            )([getattr(callbacks, cb)(**kw) for cb, kw in self.config.callbacks.items()])

    def _write_summary(self, **kwargs):
        if self.summary_writer:
            # for key, val in self.eval_dict.items():
            tags = {'F_pop': self.obj.pop,
                    'F_off': self.obj.off,
                    'F_opt': self.obj.opt}
            for key, val in tags.items():
                self.summary_writer.add_scalars(
                    main_tag=key,
                    tag_scalar_dict={
                        'mean': val.get('F').mean(),
                        'std': val.get('F').std(),
                        'max': val.get('F').max(),
                        'min': val.get('F').min()
                    },
                    global_step=self.obj.n_gen
                )
            
            self.summary_writer.close()

    def _initialize(self, **kwargs):
        super()._initialize(**kwargs)

        self.obj = copy.deepcopy(self.algorithm)
        prepare_seed(self.config.exp_cfg.seed)
        self.obj.setup(self.problem, 
                       termination=self.termination, 
                       seed=self.config.exp_cfg.seed,
                       save_history=False)
        if 'checkpoint' in self.config:
            self._load_checkpoint(f=self.config.checkpoint, **kwargs)

    def solve(self, **kwargs):
        try:
            self._initialize(**kwargs)
            self.callback_handler.begin_fit(agent=self, **kwargs)
            while self.obj.has_next():
                self.callback_handler.begin_next(**kwargs)
                self.obj.next()
                self._write_summary(**kwargs)
                self.callback_handler.after_next(**kwargs)
            self._finalize(**kwargs)

        except KeyboardInterrupt:
            self.logger.info('You have entered CTRL+C... Wait to finalize')
            self._finalize(**kwargs)
        except Exception as e:
            self.logger.error(e, exc_info=True)
            self._finalize(**kwargs)

    def _finalize(self, **kwargs):
        result = self.obj.result()
        result.problem = None
        
        super()._save_checkpoint(api=torch, 
                                 obj=result, 
                                 f=os.path.join(self.config.out_dir, 'result.pth.tar'))

    def _load_checkpoint(self, **kwargs):
        try:
            checkpoint = \
                super()._load_checkpoint(api=torch, **kwargs)
        except:
            self.logger.warn('Checkpoint not found, proceed algorithm from scratch!')
            return

        self.obj = checkpoint['obj']
        self.obj.problem = self.problem
        self.obj.problem.score_dict = checkpoint['score_dict']
        self.obj.problem.elitist_archive = checkpoint['elitist_archive']

        return checkpoint

    def _save_checkpoint(self, checkpoint={}, **kwargs):        
        problem = self.obj.problem
        self.obj.problem = None
        checkpoint['elitist_archive'] = getattr(problem, 'elitist_archive', None)
        checkpoint['score_dict'] = getattr(problem, 'score_dict', None)
        checkpoint['obj'] = copy.deepcopy(self.obj) 
        if self.obj.n_gen == 1:
            checkpoint['algorithm'] = self.algorithm
        filepath = '[{}_{}] G-{}.pth.tar'.format(
            self.obj.__class__.__name__,
            self.problem.__class__.__name__,
            self.obj.n_gen
        )
        super()._save_checkpoint(api=torch, 
                                 obj=checkpoint, 
                                 f=os.path.join(self.config.checkpoint_dir, filepath),
                                 pickle_protocol=5)

        self.obj = checkpoint['obj']
        self.obj.problem = problem

    

        
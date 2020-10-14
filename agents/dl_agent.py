
from agents import Agent
from graphs.models import *
from datasets import *

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch
import os



class DeepLearningAgent(Agent):
    def __init__(self,
                 model,
                 model_args,
                 data_loader,
                 data_loader_args,
                 optimizer,
                 optimizer_args,
                 criterion_args,
                 criterion,
                 mode='train',
                 seed=1,
                 cuda=False,
                 max_epochs=1,
                 validate_every=None,
                 verbose=False, 
                 scheduler=None,
                 scheduler_args={},
                 grad_clip=None,
                 report_freq=1, 
                 summary_writer=False,
                 checkpoint_file=None,
                 callback=None,
                 **kwargs):
        # save important parameter
        self.scheduler = scheduler
        self.data_info = data_loader_args
        self.mode = mode
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.report_freq = report_freq
        self.validate_every = validate_every if validate_every else self.max_epochs
        self.grad_clip = grad_clip
        
        self.criterion = getattr(nn, criterion, None)(**criterion_args)

        self.model = globals()[model](**model_args)
        self.parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = getattr(optim, optimizer, None)(self.parameters,
                                                         **optimizer_args)

        if scheduler:
            self.scheduler = getattr(lr_scheduler, scheduler)(optimizer=self.optimizer, 
                                                              **scheduler_args)

        data_loader = globals()[data_loader](**data_loader_args)
        self.train_queue = data_loader.train_loader
        self.valid_queue = data_loader.test_loader

        # initialize counter
        self.current_epoch = 1
        self.current_iter = 1

        # set cuda flag
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda and not cuda:
            print("WARNING: You have a CUDA device, so you should enable CUDA")

        self.cuda = self.has_cuda and cuda

        # get device
        self.device = device = torch.device("cuda:0" if self.cuda else "cpu")
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        
        print("Program will run on *****{}*****".format(self.device))

        # set manual seed
        self.manual_seed = seed
        torch.manual_seed(self.manual_seed)
        # load checkpoint
        self.load_checkpoint(checkpoint_file) if checkpoint_file else ... 

        # summary writer
        self.summary_writer = SummaryWriter() if summary_writer else None

        # save path
        self.save_path = './pretrained_weights'

        # default messages
        self.validate_msg = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'
        self.train_msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'

        callback(self) if callback else ...

    @staticmethod
    def load_checkpoint(model_path):
        pass

    @staticmethod
    def save_checkpoint(model, 
                        model_path, 
                        model_name):
        torch.save(model.state_dict(), 
                   os.path.join(model_path, '{}.pt'.format(model_name)))

    def run(self):
        try:
            self.train() if self.mode == 'train' else self.validate()
        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize") 

    def train(self):
        best_err = valid_err = 100
        while self.current_epoch < self.max_epochs:
            self.train_one_epoch()
            self.scheduler.step() if self.scheduler else ...
            if self.current_epoch % self.validate_every == 0:
                valid_err, _ = self.validate()
            
            if valid_err < best_err:
                best_err = valid_err
                self.save_checkpoint(self.model, 
                                     self.save_path, 
                                     model_name='{}-Ep_{}-Err_{}'.format(self.model.__name__,
                                                                         self.current_epoch,
                                                                         valid_err))

        torch.cuda.empty_cache()


    def train_one_epoch(self):
        self.model.train()
        correct = total = train_loss = 0
        n_inputs = len(self.train_queue.dataset)
        for step, (inputs, targets) in enumerate(self.train_queue):
            targets, loss, outputs = self.feed_forward(inputs, targets)

            train_loss += loss.item()
            predicted = self.predict(outputs)
            total += targets.size(0)
            correct += predicted.eq(targets.view_as(predicted)).sum().item()
            
            if self.verbose and step % self.report_freq == 0:
                percentage = 100.*total/n_inputs
                print(self.train_msg.format(self.current_epoch, total, n_inputs, percentage, loss.item()))

            self.current_iter += 1
        
        avg_loss = train_loss/total
        err = 100.*(1- (correct/total))
        if self.summary_writer:
            self.summary_writer.add_scalar('Loss/train', avg_loss, self.current_epoch)
            self.summary_writer.add_scalar('Error_rate/train', err, self.current_epoch)

        self.current_epoch += 1
        return err, avg_loss

    def feed_forward(self, inputs, targets):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), 
                                     self.grad_clip)
        self.optimizer.step()

        return targets, loss, outputs

    def validate(self):
        self.model.eval()
        test_loss = correct = total = 0

        with torch.no_grad():
            for step, (inputs, targets) in enumerate(self.valid_queue):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                test_loss += self.criterion(outputs, targets).item()  # sum up batch loss

                predicted = self.predict(outputs)
                total += targets.size(0)
                correct += predicted.eq(targets.view_as(predicted)).sum().item()

            avg_loss = test_loss/total
            err = 100.*(1- (correct/total))
            if self.verbose:
                acc = 100.*correct/total
                print(self.validate_msg.format(avg_loss, correct, total, acc))

        
        if self.summary_writer:
            self.summary_writer.add_scalar('Loss/test', avg_loss, self.current_epoch)
            self.summary_writer.add_scalar('Error_rate/test', err, self.current_epoch)

        return err, avg_loss

    def predict(self, outputs):
        if outputs.shape[1] == 1:
            pred = outputs
            pred[pred <= 0.5] = 0
            pred[pred > 0.5] = 1
            return pred
        return outputs.max(dim=1, keepdims=True)[1]

    def softmax_prediction(self, outputs):
        return outputs.max(dim=1, keepdims=True)[1]
    

    def binary_prediction(self, outputs):
        pred = outputs
        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 1
        return pred

    def finalize(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, '{}.pth.tar'.format()))
from agents.agent import Agent
from graphs.models import *
from datasets import *

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter



class DeepLearningAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

        self.model = globals()[config.model](config.model_arguments)
        self.criterion = getattr(nn, config.criterion, None)()
        self.optimizer = getattr(optim, config.optimizer, None)(self.model.parameters(),
                                                                lr=config.learning_rate,
                                                                momentum=config.momentum,
                                                                weight_decay=config.weight_decay)
        self.data_loader = globals()[config.data_loader](config)

        # initialize counter
        self.current_epoch = 1
        self.current_iter = 1
        self.best_metric = 0

        # set cuda flag
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda and not self.config.cuda:
            print("WARNING: You have a CUDA device, so you should enable CUDA")

        self.cuda = self.has_cuda and self.config.cuda

        # get device
        self.device = device = torch.device("cuda:0" if self.cuda else "cpu")
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        
        print("Program will run on *****{}*****".format(self.device))

        # set manual seed
        self.manual_seed = config.seed

        # load checkpoint
        # self.load_checkpoint(self.config.checkpoint_file)

        # summary writer
        self.summary_writer = SummaryWriter()

    def run(self):
        try:
            self.train()

        except KeyboardInterrupt:
            print("You have entered CTRL+C.. Wait to finalize") 

    def train(self):
        for epoch in range(self.config.max_epochs):
            self.train_one_epoch()
            # self.validate()

            self.current_epoch += 1

    def train_one_epoch(self):
        self.model.train()
        n_trained = 0
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            n_trained += len(data)

            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()

            output = self.model(data).view_as(target)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, 
                    n_trained, 
                    len(self.data_loader.train_loader.dataset),
                    100. * (n_trained / len(self.data_loader.train_loader.dataset)), 
                    loss.item()))

            self.current_iter += 1

    def validate(self):
        self.model.eval()
        test_loss = correct = 0

        with torch.no_grad():
            for data, target in self.data_loader.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += self.criterion(output, target, size_average=False).item()  # sum up batch loss
                _, pred = output.max(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.data_loader.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, 
            correct, 
            len(self.data_loader.test_loader.dataset),
            100. * correct / len(self.data_loader.test_loader.dataset)))

    def finalize(self):
        pass

        


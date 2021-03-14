# from agents import Agent
# from graphs.models import *
# from datasets import *

# import torch
# import torch.nn as nn
# import torch.backends.cudnn as cudnn
# import torch.optim.lr_scheduler as lr_scheduler
# from torch import optim
# from torch.utils.tensorboard import SummaryWriter

# import os

# import datetime

# import pickle5 as pickle


# class DeepLearningAgent(Agent):
#     def __init__(self,
#                  config,
#                  model,
#                  data_loader,
#                  optimizer,
#                  criterion,
#                  scheduler=None,
#                  mode='train',
#                  seed=1,
#                  cuda=False,
#                  save_agent=True,
#                  save_threshold=10,
#                  empty_cache=False,
#                  max_epochs=1,
#                  validate_every=None,
#                  verbose=False,
#                  deterministic=False, 
#                  grad_clip=None,
#                  report_freq=1, 
#                  summary_writer=False,
#                  checkpoint_file=None,
#                  save_path='./pretrained_weights',
#                  callback=None,
#                  **kwargs):
#         super().__init__(config)

#         # set cuda flag
#         has_cuda = torch.cuda.is_available()
#         if has_cuda and not self.config.cuda:
#             self.logger.warning("CUDA device is available, but not being used")
#         self.cuda = has_cuda and self.config.cuda

#         # set manual seed
#         torch.manual_seed(self.config.seed)
#         if self.cuda:
#             cudnn.enabled = True
#             cudnn.benchmark = not deterministic
#             cudnn.deterministic = deterministic
#             if deterministic:
#               self.logger.info('Applying deterministic mode; cudnn disabled!')
        
#         # save important parameter
#         self.mode = mode
#         self.max_epochs = max_epochs
#         self.verbose = verbose
#         self.report_freq = report_freq
#         self.validate_every = validate_every
#         self.grad_clip = grad_clip
#         self.scheduler = scheduler
#         self.empty_cache = empty_cache
#         self.save_threshold = save_threshold
#         self.save_path = save_path


#         # initialize counter
#         self.current_epoch = self.current_iter = 1
        
#         self.criterion = getattr(
#             nn, self.config.criterion.name, None
#         )(**self.config.criterion.args)

#         self.model = globals()[self.config.model.name](**self.config.model.args)

#         self.parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))

#         self.optimizer = getattr(
#             optim, self.config.optimizer.name, None
#         )(self.parameters, **self.config.optimizer.args)

#         # get device
#         self.device = torch.device("cuda:0" if self.cuda else "cpu")
#         self.logger.info("Program will run on *****{}*****".format(self.device))
#         self.model = self.model.to(self.device)
#         self.criterion = self.criterion.to(self.device)

#         # load checkpoint
#         self.load_checkpoint(self.config.checkpoint_file)

#         if hasattr(self.config, 'scheduler'):
#             self.scheduler = getattr(
#                 lr_scheduler, self.config.scheduler.name
#             )(optimizer=self.optimizer, **self.config.scheduler.args)

#         # save agent
#         self.save_agent() if save_agent else ...
                
#         data_loader = globals()[self.config.data_loader.name](**self.config.data_loader.args)
#         self.train_queue = data_loader.train_loader
#         self.valid_queue = data_loader.test_loader

#         # summary writer
#         self.summary_writer = SummaryWriter(self.save_path) if summary_writer else None

#         # default messages
#         self.validate_msg = '{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'
#         self.train_msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'

#         callback(self) if callback else ...

#     def save_agent(self):
#         now = datetime.datetime.now()
#         name = '{}_{}.pickle'.format(self.__class__.__name__, now.strftime("%Y%m%d-%H%M"))
#         with open(os.path.join(self.save_path, name), 'wb') as handle:
#             pickle.dump(self, handle, pickle.HIGHEST_PROTOCOL)

#     def load_checkpoint(self, path=None):
#         if not path:
#           return None
#         checkpoint = torch.load(path, map_location=self.device)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         self.current_epoch = checkpoint['epoch'] + 1
#         self.criterion = checkpoint['loss']
#         self.save_threshold = checkpoint['save_threshold']
            

#     def save_checkpoint(self, error_rate, best_err):
#         checkpoint = {"epoch": self.current_epoch,
#                       "model_state_dict": self.model.state_dict(),
#                       "optimizer_state_dict": self.optimizer.state_dict(),
#                       "loss": self.criterion,
#                       "save_threshold": best_err}

#         name = '{}-Ep_{:03d}-Err_{:.3f}.pth.tar'
#         filepath = os.path.join(self.save_path, name.format(self.model.__class__.__name__,
#                                                             self.current_epoch,
#                                                             error_rate))
#         torch.save(checkpoint, filepath)
        
#     def run(self):
#         try:
#             self.train() if self.mode == 'train' else self.validate()
#         except KeyboardInterrupt:
#             print("You have entered CTRL+C.. Wait to finalize") 

#     def train(self):
#         best_err = valid_err = self.save_threshold
#         while self.current_epoch <= self.max_epochs:
#             self.train_one_epoch()
#             if self.scheduler:
#                 self.scheduler.step()
#                 print('Epoch: {} - lr {}'.format(self.current_epoch, 
#                                                  self.scheduler.get_last_lr()[0]))
#             if self.current_epoch % self.validate_every == 0:
#                 valid_err, _ = self.validate()
            
#             if valid_err < best_err:
#                 best_err = valid_err
#                 self.save_checkpoint(valid_err, best_err)

#         if self.empty_cache:
#           torch.cuda.empty_cache()


#     def train_one_epoch(self):
#         self.model.train()
#         correct = total = train_loss = 0
#         n_inputs = len(self.train_queue.dataset)
#         for step, (inputs, targets) in enumerate(self.train_queue):
#             targets, loss, outputs = self.feed_forward(inputs, targets)

#             train_loss += loss.item()
#             predicted = self.predict(outputs)
#             total += targets.size(0)
#             correct += predicted.eq(targets.view_as(predicted)).sum().item()
            
#             if self.verbose and step % self.report_freq == 0:
#                 percentage = 100.*total/n_inputs
#                 print(self.train_msg.format(self.current_epoch, 
#                                             total, 
#                                             n_inputs, 
#                                             percentage, 
#                                             loss.item()))

#             self.current_iter += 1
        
#         avg_loss = train_loss/total
#         err = 100.*(1- (correct/total))
#         if self.verbose:
#             acc = 100.*correct/total
#             print(self.validate_msg.format('Train', avg_loss, correct, total, acc))
#         if self.summary_writer:
#             self.summary_writer.add_scalar('Loss/train', avg_loss, self.current_epoch)
#             self.summary_writer.add_scalar('Error_rate/train', err, self.current_epoch)

#         self.current_epoch += 1
#         return err, avg_loss

#     def feed_forward(self, inputs, targets):
#         inputs, targets = inputs.to(self.device), targets.to(self.device)
#         self.optimizer.zero_grad()
#         outputs = self.model(inputs)
#         loss = self.criterion(outputs, targets)

#         loss.backward()
#         if self.grad_clip:
#             nn.utils.clip_grad_norm_(self.model.parameters(), 
#                                      self.grad_clip)
#         self.optimizer.step()

#         return targets, loss, outputs

#     def validate(self):
#         self.model.eval()
#         test_loss = correct = total = 0

#         with torch.no_grad():
#             for step, (inputs, targets) in enumerate(self.valid_queue):
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)

#                 outputs = self.model(inputs)
#                 test_loss += self.criterion(outputs, targets).item()  # sum up batch loss

#                 predicted = self.predict(outputs)
#                 total += targets.size(0)
#                 correct += predicted.eq(targets.view_as(predicted)).sum().item()

#             avg_loss = test_loss/total
#             err = 100.*(1- (correct/total))
#             if self.verbose:
#                 acc = 100.*correct/total
#                 print(self.validate_msg.format('Test', avg_loss, correct, total, acc))

        
#         if self.summary_writer:
#             self.summary_writer.add_scalar('Loss/test', avg_loss, self.current_epoch)
#             self.summary_writer.add_scalar('Error_rate/test', err, self.current_epoch)

#         return err, avg_loss

#     def predict(self, outputs):
#         _, pred = outputs.max(1)
#         return pred

#     def finalize(self):
#         now = datetime.datetime.now()
#         torch.save(self.model.state_dict(), os.path.join(self.save_path, '{}.pth.tar'.format(now.strftime("%Y%m%d-%H%M"))))
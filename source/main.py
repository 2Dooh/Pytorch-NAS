
import torch
from torch import optim, nn
import time
from utils.transform import *
from graphs.models import LogisticRegression, FullyConnected
import pandas as pd
import matplotlib.pyplot as plt
from datasets import LoanPrediction
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Feature Engineering
# transform = transforms.Compose([LoanPredictionToNumeric(), 
#                                 FeatureNormalizer(),
#                                 transforms.ToTensor(),
#                                 Squeeze()])
# train_set = transform(pd.read_csv('./data/loan_prediction/train_u6lujuX_CVtuZ9i.csv'))
# train_set = Dataset(X=train_set[:, :-1], Y=train_set[:, -1])

# # Create dataloader objects
# train_loader = DataLoader(dataset=train_set,
#                           batch_size=128,
#                           shuffle=True)
train_loader = LoanPrediction(json.load('./configs/loan_prediction.json'))
# val_loader = DataLoader(dataset=val_set, batch_size=len(val_set))

# Hyperparameter settings
learning_rate = 0.01

model = LogisticRegression(input_size=11, 
                          activation=torch.relu)
# model = FullyConnected([11, 100, 100, 100, 1], [torch.relu, torch.relu, torch.relu, None])
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), learning_rate)


def train(model,
          optimizer,
          criterion,

          train_loader,
          epochs):
    # switch to train mode
    LOSS = [None] * epochs
    model.train()
    model.to(device)
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()

            LOSS[epoch] = loss
        print(LOSS[epoch])

    return LOSS


def validate(model,
             criterion,

             val_loader):
    # switch to evaluate mode
    model.eval()
    correct = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            z = model(x)
            _, y_pred = torch.max(z.data, dim=1)
            correct += (y_pred == y).sum().item()
    accuracy = correct / len(val_loader)

    return accuracy

torch.manual_seed(1)
loss = train(model,
      optimizer,
      criterion,
      train_loader,
      epochs=500)

n_iters = torch.arange(len(loss))
plt.plot(n_iters, loss)
plt.grid(True, linestyle='--')
plt.legend(loc='upper right')
plt.xlabel('Iterations/Epochs')
plt.ylabel('Cost')
plt.show()
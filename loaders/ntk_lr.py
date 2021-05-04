from torch.utils.data import DataLoader, TensorDataset, Dataset

from torchvision import transforms

import torch

import pandas as pd

import os

from sklearn.model_selection import train_test_split

import numpy as np

class NTK_LR_Regression:
    def __init__(self, 
                 path, 
                 batch_size, 
                 test_batch_size, 
                 data_loader_workers, 
                 pin_memory,
                 **kwargs):

        data_path = os.path.join(path)
        self.raw_data = pd.read_csv(data_path)
        dataset = self.raw_data.values[:, 1:]

        X, y = np.hsplit(dataset, [2])

        # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        # for train, test in sss.split(data, labels):
        #     train_data, train_labels = data[train], labels[train]
        #     test_data, test_labels = data[test], labels[test]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        train = TensorDataset(torch.from_numpy(X_train).type(torch.float), torch.from_numpy(y_train).type(torch.float))
        test = TensorDataset(torch.from_numpy(X_test).type(torch.float), torch.from_numpy(y_test).type(torch.float))

        self.train_loader = DataLoader(dataset=train, 
                                       batch_size=batch_size, 
                                       shuffle=True,
                                       num_workers=data_loader_workers,
                                       pin_memory=pin_memory)

        self.test_loader = DataLoader(dataset=test,
                                      batch_size=test_batch_size,
                                      shuffle=False,
                                      num_workers=data_loader_workers,
                                      pin_memory=pin_memory)






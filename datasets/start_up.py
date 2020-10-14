from torch.utils.data import TensorDataset, DataLoader, Dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
import numpy as np

class StartUp:
    def __init__(self,
                 data_folder,
                 pin_memory,
                 batch_size,
                 test_batch_size,
                 num_workers,
                 train_set,
                 test_set,
                 **kwargs):
        
        train_path = os.path.join(data_folder, train_set)
        test_path = os.path.join(data_folder, test_set)
        
        train_set = self.__transform(pd.read_csv(train_path))
        test_set = self.__transform(pd.read_csv(test_path))

        # X_train, X_test, y_train, y_test = \
        #     train_test_split(dataset[:, :-1],
        #                      dataset[:, -1][:, np.newaxis],
        #                      train_size=0.8,
        #                      random_state=0,
        #                      shuffle=False)

        train = TensorDataset(train_set[:, :-1], train_set[:, -1].view(-1, 1))
        test = TensorDataset(test_set[:, :-1], test_set[:, -1].view(-1, 1))

        self.train_loader = DataLoader(dataset=train,
                                       shuffle=False,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory)

        self.test_loader = DataLoader(dataset=test,
                                      shuffle=False,
                                      batch_size=test_batch_size,
                                      num_workers=num_workers,
                                      pin_memory=pin_memory)

    @staticmethod
    def __transform(samples):
        state = samples['State']
        state = LabelEncoder().fit(state).transform(state)[:, np.newaxis]
        state = OneHotEncoder().fit(state).transform(state).toarray().astype(np.float)
        no_state_df = samples.drop(columns=['State'], axis=1).values[:, 1:]
        dataset = np.hstack((state, no_state_df))
        X = dataset[:, :-1]
        mu, sigma = X.mean(axis=0), X.max(axis=0) - X.min(axis=0)
        dataset[:, :-1] = ((X - mu) / sigma).astype(np.float)
        tensor_dataset = torch.from_numpy(dataset).type(torch.float)
        return tensor_dataset
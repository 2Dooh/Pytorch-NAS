from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np

class Salary:
    def __init__(self, 
                 data_folder, 
                 train_set, 
                 batch_size,
                 **kwargs):
        
        data_path = os.path.join(data_folder, train_set)
        self.dataframe = pd.read_csv(data_path)
        dataset = torch.tensor(self.dataframe.values, dtype=torch.float)

        X_train, X_test, y_train, y_test = \
            train_test_split(dataset[:, :-1], 
                             dataset[:, -1],
                             train_size=0.8,
                             random_state=0,
                             shuffle=False)

        
        train = TensorDataset(X_train, y_train)
        test = TensorDataset(X_test, y_test)

        self.train_loader = DataLoader(dataset=train, shuffle=True, batch_size=batch_size)
        self.test_loader = DataLoader(dataset=test, shuffle=False)
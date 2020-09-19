from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
import torch
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

class LoanPrediction:
    def __init__(self, config):
        self.config = config

        data_path = os.path.join(config.data_folder, config.train_set)
        dataset = pd.read_csv(data_path)

        dataset = self.__transform(dataset)
        
        data, labels = dataset[:, :-1], dataset[:, -1]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train, test in sss.split(data, labels):
            train_data, train_labels = data[train], labels[train]
            test_data, test_labels = data[test], labels[test]

        train = TensorDataset(train_data, train_labels)
        test = TensorDataset(test_data, test_labels)

        self.train_loader = DataLoader(dataset=train, 
                                       batch_size=self.config.batch_size, 
                                       shuffle=True,
                                       num_workers=self.config.data_loader_workers,
                                       pin_memory=self.config.pin_memory)

        self.test_loader = DataLoader(dataset=test,
                                      batch_size=config.test_batch_size,
                                      shuffle=False,
                                      num_workers=self.config.data_loader_workers,
                                      pin_memory=self.config.pin_memory)

    
    
    def __transform(self, dataset):
        class ToNumeric(object):
            def __call__(self, sample):
                

                return new_sample.values

        class Normalization(object):
            def __call__(self, sample):
                new_sample = sample.squeeze().type(torch.float)
                mu = new_sample[:, :-1].mean(axis=0)
                sigma = new_sample[:, :-1].std(axis=0)   

                new_sample[:, :-1] = (new_sample[:, :-1] - mu) / sigma

                return new_sample
        transform = transforms.Compose([ToNumeric(),
                                        transforms.ToTensor(),
                                        Normalization()])
        return transform(dataset)




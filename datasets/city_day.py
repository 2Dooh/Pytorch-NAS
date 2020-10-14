from torch.utils.data import TensorDataset, DataLoader, Dataset
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
import numpy as np

class CityDay:
    def __init__(self,
                 data_folder,
                 pin_memory,
                 batch_size,
                 test_batch_size,
                 num_workers,
                 train_set,
                 test_set,
                 **kwargs):
        super().__init__()

        train_path = os.path.join(data_folder, train_set)
        test_path = os.path.join(data_folder, test_set)

        train_set = self.__transform(pd.read_csv(train_path))
        test_set = self.__transform(pd.read_csv(test_path))

        train = TensorDataset(train_set[:, :-1], train_set[:, -1].view(-1, 1))
        test = TensorDataset(test_set[:, :-1], test_set[:, -1].view(-1, 1))

        self.train_loader = DataLoader(dataset=train,
                                       shuffle=True,
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
        samples.drop('Date', axis=1, inplace=True)
        label_features = list(filter(lambda c : samples[c].dtype == pd.CategoricalDtype, samples.columns.values))
        num_features = list(set(samples.columns.values) - set(label_features))

        cat_data = samples[label_features]
        cat_data = cat_data.apply(lambda x : x.fillna(x.value_counts().index[0]))
        le = LabelEncoder()
        le_data = cat_data.apply(le.fit_transform)
        enc = OneHotEncoder()
        cat_data = enc.fit(le_data).transform(le_data).toarray()

        num_data = samples[num_features].copy()
        num_data.interpolate(inplace=True)
        target = num_data['AQI']
        num_data.drop('AQI', axis=1, inplace=True)

        dataset = np.concatenate([cat_data, num_data.values, target.values[:, np.newaxis]], axis=1)
        dataset = (dataset - dataset.mean(axis=0)) / dataset.std(axis=0)
        tensor_dataset = torch.from_numpy(dataset).type(torch.float)
        return tensor_dataset
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
import torch
import pandas as pd
import os

class LoanPrediction:
    def __init__(self, config):
        self.config = config

        train_path = os.path.join(config.data_folder, config.train_set)
        valid_path = os.path.join(config.data_folder, config.test_set)
        train_set = pd.read_csv(train_path)
        valid_set = pd.read_csv(valid_path)

        train_set = self.__transform(train_set)
        valid_set = self.__transform(valid_set)
        
        train_data, train_labels = train_set[:, :-1], train_set[:, -1]
        valid_data, valid_labels = valid_set[:, :-1], valid_set[:, -1]

        train = TensorDataset(train_data, train_labels)
        valid = TensorDataset(valid_data, valid_labels)

        self.train_loader = DataLoader(train, 
                                       batch_size=self.config.batch_size, 
                                       shuffle=True,
                                       num_workers=self.config.data_loader_workers,
                                       pin_memory=self.config.pin_memory)

        self.test_loader = DataLoader(valid,
                                      batch_size=config.test_batch_size,
                                      shuffle=False,
                                      num_workers=self.config.data_loader_workers,
                                      pin_memory=self.config.pin_memory)

    
    
    def __transform(self, dataset):
        class ToNumeric(object):
            def __call__(self, sample):
                gender = {'Male': 1, 'Female': 0}
                married = {'Yes': 1, 'No': 0}
                dependents = {'0': 0, '1': 1, '3+': 2}
                education = {'Graduate': 1, 'Not Graduate': 0}
                self_employed = {'Yes': 1, 'No': 0}
                property_area = {'Rural': 0, 'Urban': 2, 'Semiurban': 1}
                loan_status = {'Y': 1, 'N': 0}


                new_sample = sample.drop(['Loan_ID'], axis=1)

                new_sample['Gender'] = new_sample['Gender'].map(gender)
                new_sample['Married'] = new_sample['Married'].map(married)
                new_sample['Dependents'] = new_sample['Dependents'].map(dependents)
                new_sample['Education'] = new_sample['Education'].map(education)
                new_sample['Self_Employed'] = new_sample['Self_Employed'].map(self_employed)
                new_sample['Property_Area'] = new_sample['Property_Area'].map(property_area)
                if 'Loan_Status' in new_sample.columns:
                    new_sample['Loan_Status'] = new_sample['Loan_Status'].map(loan_status)
                
                new_sample = new_sample.dropna()

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




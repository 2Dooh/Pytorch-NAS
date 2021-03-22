from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
import torch
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import numpy as np

class LoanPrediction:
    def __init__(self, 
                 data_folder, 
                 train_set, 
                 batch_size, 
                 test_batch_size, 
                 data_loader_workers, 
                 pin_memory,
                 **kwargs):

        data_path = os.path.join(data_folder, train_set)
        self.dataframe = pd.read_csv(data_path)

        dataset = self.__transform(self.dataframe)
        
        data, labels = dataset[:, :-1], dataset[:, -1]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train, test in sss.split(data, labels):
            train_data, train_labels = data[train], labels[train]
            test_data, test_labels = data[test], labels[test]

        train = TensorDataset(train_data, train_labels)
        test = TensorDataset(test_data, test_labels)

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

    
    
    def __transform(self, dataset):
        class ToNumeric(object):
            def __call__(self, sample):
                gender = {'Male': 1, 'Female': 0}
                married = {'Yes': 1, 'No': 0}
                dependents = {'0': 0, '1': 1, '3+': 2}
                education = {'Graduate': 1, 'Not Graduate': 0}
                self_employed = {'Yes': 1, 'No': 0}
                property_area = {'Rural': 0, 'Urban': 2, 'Semiurban': 1}
                loan_status = {'Y': 0, 'N': 1}


                new_sample = sample.drop(['Loan_ID'], axis=1)

                new_sample['Gender'] = new_sample['Gender'].map(gender)
                new_sample['Married'] = new_sample['Married'].map(married)
                new_sample['Dependents'] = new_sample['Dependents'].map(dependents)
                new_sample['Education'] = new_sample['Education'].map(education)
                new_sample['Self_Employed'] = new_sample['Self_Employed'].map(self_employed)
                new_sample['Property_Area'] = new_sample['Property_Area'].map(property_area)
                new_sample['Loan_Status'] = new_sample['Loan_Status'].map(loan_status)

                new_sample = new_sample.dropna()


                new_sample['ApplicantIncome'] = new_sample['CoapplicantIncome'] / new_sample['ApplicantIncome']
                new_sample['LoanAmount'] = np.log(new_sample['LoanAmount'] * new_sample['Loan_Amount_Term'])


                new_sample.drop(['CoapplicantIncome', 'Loan_Amount_Term'], axis=1, inplace=True)

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




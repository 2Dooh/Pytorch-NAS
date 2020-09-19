from torch.utils.data import Dataset
import torch
import os
import pandas as pd
import cv2 as cv

class Imageset(Dataset):
    def __init__(self, 
                 csv_file, 
                 data_dir,
                 transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        csv_file_dir = os.path.join(self.data_dir, csv_file)

        self.data_name = pd.read_csv(csv_file_dir)
        self.len = self.data_name.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        name = os.path.join(self.data_dir, self.data_name[idx, 1])
        x = cv.imread(name)
        y = self.data_name[idx, 0]

        x = self.transform(x) if self.transform is not None else x

        return x, y
 
class Dataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.len = len(X)

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        return x, y


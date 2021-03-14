from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch

class Mnist:
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    def __init__(self,
                 data_folder,
                 num_workers,
                 batch_size,
                 pin_memory,
                 **kwargs):
        
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(self.MNIST_MEAN, self.MNIST_STD),
                                        torch.flatten])

        train_data = datasets.MNIST(root=data_folder,
                                    train=True,
                                    download=True,
                                    transform=transform)
        test_data = datasets.MNIST(root=data_folder,
                                   train=False,
                                   download=False,
                                   transform=transform)
        
        self.train_loader = DataLoader(dataset=train_data,
                                        batch_size=batch_size,
                                        pin_memory=pin_memory,
                                        num_workers=num_workers,
                                        shuffle=True)
        self.test_loader = DataLoader(dataset=test_data,
                                        batch_size=batch_size,
                                        pin_memory=pin_memory,
                                        num_workers=num_workers,
                                        shuffle=False)
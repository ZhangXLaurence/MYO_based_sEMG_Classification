import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.utils.data

import os

from . import CSVDataset
from . import DataTransformCompose

# Load MNIST
def LoadMNIST(train_batch_size, test_batch_size, path):
    trainset = datasets.MNIST(path, download=True, train=True, transform=DataTransformCompose.TransformMNIST())
    train_loader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=4)
    testset = datasets.MNIST(path, download=True, train=False, transform=DataTransformCompose.TransformMNIST())
    test_loader = DataLoader(testset, test_batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader

# Load Fashion-MNIST
def LoadFashionMNIST(train_batch_size, test_batch_size, path):
    trainset = datasets.FashionMNIST(path, download=True, train=True, transform=DataTransformCompose.TransformMNIST())
    train_loader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=4)
    testset = datasets.FashionMNIST(path, download=True, train=False, transform=DataTransformCompose.TransformMNIST())
    test_loader = DataLoader(testset, test_batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader

# Load CSV
def LoadCSV(train_batch_size, test_batch_size, cvs_file_dir):
    trainset = CSVDataset(cvs_file_dir)
    # train_loader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=4)
    train_loader = DataLoader(trainset, train_batch_size, sampler=ImbalancedDatasetSampler(trainset), shuffle=True, num_workers=4)
    testset = CSVDataset(cvs_file_dir)
    test_loader = DataLoader(testset, test_batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader

# Load CIFAR10
def LoadCIFAR10(train_batch_size, test_batch_size, path, arg_inputsize=224):
    trainset = datasets.CIFAR10(path, download=True, train=True, transform=DataTransformCompose.TransformCIFAR10(True, arg_inputsize))
    train_loader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=4)
    testset = datasets.CIFAR10(path, download=True, train=False, transform=DataTransformCompose.TransformCIFAR10(False, arg_inputsize))
    test_loader = DataLoader(testset, test_batch_size, shuffle=True, num_workers=4)
    return train_loader, test_loader


#######################################



class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is datasets.ImageFolder:
            return dataset.imgs[idx][1]
        elif dataset_type is CSVDataset:
            # print(dataset.GetLabel(idx))
            return dataset.GetLabel(idx)
        else:
            raise NotImplementedError
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
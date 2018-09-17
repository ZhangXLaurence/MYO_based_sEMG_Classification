import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
    train_loader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=4)
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

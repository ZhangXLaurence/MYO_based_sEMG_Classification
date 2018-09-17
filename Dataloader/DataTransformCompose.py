import torch
from torchvision import transforms


##################################    MNIST Dataset     ##################################
def TransformMNIST():
    return transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
##################################    End MNIST Dataset     ##################################


##################################    CIFAR-10 Dataset     ##################################
def TransformCIFAR10(isTrain=True, arg_InputSize=224):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if isTrain:
        transform_train = transforms.Compose([
            # transforms.RandomCrop(32, padding=4),
            transforms.Resize(arg_InputSize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,])
        return transform_train
    else:
        transform_test = transforms.Compose([
            transforms.Resize(arg_InputSize),
            transforms.ToTensor(),
            normalize,])
        return transform_test
##################################    End CIFAR-10 Dataset     ##############################

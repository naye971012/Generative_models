import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import *

def get_loaders(args) -> Tuple[DataLoader,DataLoader,Tuple]:
    """
    get specific dataset from args.data_name
    return train/test dataloader and classes
    """
    if(args.data_name=='cifar10'):
        trainset, testset, classes = get_dataset_cifar10(args)
    else:
        raise "data not exist"

    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, drop_last=True)    
    test_loader = DataLoader(testset, batch_size=args.batch_size,
                                            shuffle=False, drop_last=True)

    return train_loader, test_loader, classes


def get_dataset_cifar10(args):
    trainset = datasets.CIFAR10(root=args.data_save_path, train=True,
                                            download=True, transform=transforms.Compose([
                                                            transforms.Resize((64, 64)),  # Resize images to 64x64
                                                            transforms.ToTensor(),        # Convert images to PyTorch tensors
                                                            transforms.Normalize(         # Normalize the tensor values
                                                                mean=[0.5, 0.5, 0.5],     # Mean for each channel
                                                                std=[0.5, 0.5, 0.5]       # Standard deviation for each channel
                                                            )
                                                        ]))

    testset = datasets.CIFAR10(root=args.data_save_path, train=False,
                                        download=True, transform=transforms.Compose([
                                                        transforms.Resize((64, 64)),  # Resize images to 64x64
                                                        transforms.ToTensor(),        # Convert images to PyTorch tensors
                                                        transforms.Normalize(         # Normalize the tensor values
                                                            mean=[0.5, 0.5, 0.5],     # Mean for each channel
                                                            std=[0.5, 0.5, 0.5]       # Standard deviation for each channel
                                                        )
                                                    ]))
    
    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainset, testset, classes

def get_dataset_mnist(args):
    trainset = datasets.MNIST(root=args.save_path, train=True,
                                            download=True, transform=transforms.ToTensor())

    testset = datasets.MNIST(root=args.save_path, train=False,
                                        download=True, transform=transforms.ToTensor())
    
    classes = ()

    return trainset, testset, classes





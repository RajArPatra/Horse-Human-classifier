# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 03:23:15 2020

@author: HP
"""
import torch
import torchvision.transforms as transforms
import torchvision


def get_data(train='DATASET\\train',test='DATASET\\validation'):
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.ImageFolder(root = train, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=4)

    testset = torchvision.datasets.ImageFolder(root = test, transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=4)
    return trainloader,testloader,trainset.classes
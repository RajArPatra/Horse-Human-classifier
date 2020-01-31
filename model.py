# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 01:55:22 2020

@author: ar
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,in_fea=3,out_fea=2):
        super().__init__()
        self.pad=nn.ZeroPad2d(2)
        self.conv1=nn.Conv2d(3,8,5)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2=nn.Conv2d(8,16,5)
        self.fc1= nn.Linear(16*75*75,128)
        self.fc2= nn.Linear(128,64)
        self.fc3= nn.Linear(64,2)
        
    def forward(self,x):
        x=self.pad(x)
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=self.pad(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=x.view(-1,16*75*75)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
        
        
        
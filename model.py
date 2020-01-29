# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 01:55:22 2020

@author: HP
"""

import torch
import torchvision
import torch.nn as nn



class Net(nn.Module):
    def __init__(self,in_fea=3,out_fea=len(classes)):
        super(Net,self).__init__()
        self.pad()=nn.Zeropad(2)
        self.conv1=nn.conv2d(3,8,5)
        self.pool=nn.MaxPool2d(kernel_size=2,stide=1)
        self.con2=nn.conv2d(8,16,5)
        self.fc1= nn.Linear(16*300*300,128)
        self.fc2= nn.Linear(128,64)
        self.fc3= nn.Linear(64,out_fea)
        
    def forward(self,x):
        x=self.pad(x)
        x=nn.Relu(self.conv1(x))
        x=self.pool(x)
        x=self.pad(x)
        x=nn.Relu(self.conv2(x))
        x=self.pool(x)
        x=Relu(self.fc1(x))
        x=Relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
        
        
        
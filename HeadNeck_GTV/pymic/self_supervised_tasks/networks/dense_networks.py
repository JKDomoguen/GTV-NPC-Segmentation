# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np


class FullyConnected(nn.Module):
    def __init__(self,input_size=512,hidden_size=512, num_classes=100, dropout_rate=0.5, include_top=True):
        super(FullyConnected,self).__init__()
        

        layers = []
        layers.append(nn.Linear(input_size,hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Linear(hidden_size,hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_size))

        if include_top:
            layers.append(nn.Linear(hidden_size,num_classes))
            layers.append(nn.Softmax(dim=1))

        self._model = nn.Sequential(*layers)

    def forward(self,inputs):
        return self._model(inputs)

class SimpleMulticlass(nn.Module):
    def __init__(self,input_size=512,num_classes=5,hidden_size=1024,dropout_rate=0.5,include_top=True, **kwargs):
        super(SimpleMulticlass,self).__init__()
        layers = []
        layers.append(nn.Linear(input_size,hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.Dropout(p=dropout_rate))
        
        if include_top:
            layers.append(nn.Linear(hidden_size,num_classes))
            layers.append(nn.Softmax(dim=1))
        self._model = nn.Sequential(*layers)

    def forward(self,inputs):
        return self._model(inputs)

class FullyConnectedBig(nn.Module):
    def __init__(self,input_size=2048,hidden_size=2048, num_classes=1, dropout_rate=0.5, include_top=True):
        super(FullyConnectedBig,self).__init__()
        layers = []
        #1st Layer
        layers.append(nn.Linear(input_size,hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_size))
        layers.append(nn.Dropout(p=dropout_rate))
        #2nd Layer
        layers.append(nn.Linear(hidden_size,hidden_size//2))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_size//2))
        layers.append(nn.Dropout(p=dropout_rate))

        if include_top:
            layers.append(nn.Linear(hidden_size//2,num_classes))
            layers.append(nn.ReLU())

        self._model = nn.Sequential(*layers)

    def forward(self,inputs):
        return self._model(inputs)
        
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np


from pymic.layer.activation import get_acti_func
from pymic.layer.convolution import ConvolutionLayer


class UNet2D5Block(nn.Module):
    def __init__(self, in_channels, out_channels, 
            kernel_size, paddding, acti_func, acti_func_param):
        super(UNet2D5Block, self).__init__()
        
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func

        self.conv1 = ConvolutionLayer(in_channels,  out_channels, kernel_size = kernel_size, 
                padding = paddding, acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = ConvolutionLayer(out_channels, out_channels, kernel_size = kernel_size, 
                padding = paddding, acti_func=get_acti_func(acti_func, acti_func_param))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNet3DBlock(nn.Module):
    def __init__(self,in_channels, out_channels, acti_func, acti_func_param):
        super(UNet3DBlock, self).__init__()
        
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func

        self.conv1 = ConvolutionLayer(in_channels,  out_channels, 3, 
                padding = 1, acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = ConvolutionLayer(out_channels, out_channels, 3, 
                padding = 1, acti_func=get_acti_func(acti_func, acti_func_param))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UNet2D5Encoder(nn.Module):

    def __init__(self, params,down1=False,down2=False,down3=False):
        super(UNet2D5Encoder, self).__init__()
        self.params = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.acti_func = self.params['acti_func']
        assert(len(self.ft_chns) == 5)        

        self.block1 = UNet2D5Block(self.in_chns, self.ft_chns[0], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block2 = UNet2D5Block(self.ft_chns[0], self.ft_chns[1], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block3 = UNet2D5Block(self.ft_chns[1], self.ft_chns[2], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block4 = UNet2D5Block(self.ft_chns[2], self.ft_chns[3], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block5 = UNet2D5Block(self.ft_chns[3], self.ft_chns[4], 
            (3, 3, 3), (1, 1, 1), self.acti_func, self.params)

        if down1:
            self.down1 = nn.MaxPool3d(kernel_size = 2)
        else:                        
            self.down1 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))

        if down2:
            self.down2 = nn.MaxPool3d(kernel_size = 2)
        else:                        
            self.down2 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))

        if down3:
            self.down3 = nn.MaxPool3d(kernel_size = 2)
        else:                        
            self.down3 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))

        # self.down2 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))
        # self.down3 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))
        # self.down3 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))
        self.down4 = nn.MaxPool3d(kernel_size = 2)

    def forward(self, x):
        f1 = self.block1(x);  d1 = self.down1(f1)
        f2 = self.block2(d1); d2 = self.down2(f2)
        f3 = self.block3(d2); d3 = self.down3(f3)
        f4 = self.block4(d3); d4 = self.down4(f4)
        f5 = self.block5(d4)
        return f5

class UNet3DEncoder(nn.Module):
    def __init__(self, params):
        super(UNet3DEncoder, self).__init__()

        self.params = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.acti_func = self.params['acti_func']
        self.dropout   = self.params['dropout']
        self.resolution_level = len(self.ft_chns)

        self.block1 = UNet3DBlock(self.in_chns, self.ft_chns[0], 
             self.acti_func, self.params)

        self.block2 = UNet3DBlock(self.ft_chns[0], self.ft_chns[1], 
             self.acti_func, self.params)

        self.block3 = UNet3DBlock(self.ft_chns[1], self.ft_chns[2], 
             self.acti_func, self.params)

        self.block4 = UNet3DBlock(self.ft_chns[2], self.ft_chns[3], 
             self.acti_func, self.params)

        self.block5 = UNet3DBlock(self.ft_chns[3], self.ft_chns[4], 
                self.acti_func, self.params)

        self.down1 = nn.MaxPool3d(kernel_size = 2)
        self.down2 = nn.MaxPool3d(kernel_size = 2)
        self.down3 = nn.MaxPool3d(kernel_size = 2)
        self.down4 = nn.MaxPool3d(kernel_size = 2)

        if(self.dropout):
             self.drop1 = nn.Dropout(p=0.1)
             self.drop2 = nn.Dropout(p=0.1)
             self.drop3 = nn.Dropout(p=0.2)
             self.drop4 = nn.Dropout(p=0.2)
             self.drop5 = nn.Dropout(p=0.2)
    
    def forward(self,x):

        f1 = self.block1(x)
        if(self.dropout):
             f1 = self.drop1(f1)
        d1 = self.down1(f1)

        f2 = self.block2(d1)
        if(self.dropout):
             f2 = self.drop2(f2)
        d2 = self.down2(f2)

        f3 = self.block3(d2)
        if(self.dropout):
             f3 = self.drop3(f3)
        d3 = self.down3(f3)

        f4 = self.block4(d3)
        if(self.dropout):
             f4 = self.drop4(f4)
        d4 = self.down4(f4)

        f5 = self.block5(d4)
        if(self.dropout):
            f5 = self.drop5(f5)
        
        return torch.flatten(f5,start_dim=1)
        # return f5
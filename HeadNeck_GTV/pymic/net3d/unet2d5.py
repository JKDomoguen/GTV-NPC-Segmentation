# -*- coding: utf-8 -*-
from __future__ import print_function, division
from unicodedata import bidirectional

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from pymic.layer.activation import get_acti_func
from pymic.layer.convolution import ConvolutionLayer
from pymic.layer.deconvolution import DeconvolutionLayer

class UNetBlock3D(nn.Module):
    def __init__(self,in_channels, out_channels, acti_func, acti_func_param):
        super(UNetBlock3D, self).__init__()
        
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
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
            kernel_size, paddding, acti_func, acti_func_param):
        super(UNetBlock, self).__init__()
        
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
    

class SE_BlockRNN(nn.Module):
    def __init__(self, c,hidden_rnn=4,rnn_layer=2,r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool3d((None,1,1) )
        self.excitation = nn.Sequential(
            nn.Linear(2*hidden_rnn*c, (2*hidden_rnn*c) // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((2*hidden_rnn*c) // r, c, bias=False),
            nn.Sigmoid()
        )
        self.excitation_lstm = nn.LSTM(1,hidden_rnn,rnn_layer,batch_first=True,bidirectional=True)

    def forward(self, x):
        bs, c,d, _, _ = x.shape
        y = self.squeeze(x).view(bs*c,d,1)
        y,(_,_) = self.excitation_lstm(y)
        y = y[:,-1,:].squeeze().view(bs,-1)
        y = self.excitation(y).view(bs, c,1,1, 1)
        return x * y.expand_as(x)


class SE_BlockV2(nn.Module):
    def __init__(self, channel,depth,r=4):
        super().__init__()
        self.squeeze_depth = nn.AdaptiveAvgPool3d((None,1,1) )
        self.squeeze_channel = nn.AdaptiveAvgPool3d((1,1,1) )
        self.excitation_depth = nn.Sequential(
            nn.Linear(depth, 2*depth, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2*depth, depth, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(depth, depth // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(depth // r, depth, bias=False),
            nn.Sigmoid()
        )

        self.excitation_channel = nn.Sequential(
            nn.Linear(channel, 2*channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(2*channel, channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c,d, _, _ = x.shape
        y_depth_squeezed = self.squeeze_depth(x).view(bs,c,d)
        y_depth_excited = self.excitation_depth(y_depth_squeezed).view(bs,c,d,1,1)
        x_depth_excited = x*y_depth_excited.expand_as(x)
        y_channel_squeezed = self.squeeze_channel(x_depth_excited).view(bs,c)
        y_channel_excited = self.excitation_channel(y_channel_squeezed).view(bs,c,1,1,1)
        
        return x_depth_excited * y_channel_excited.expand_as(x)


# class ConvolutionLayerSNEDepth(nn.Module):
#     """
#     A compose layer with the following components:
#     convolution -> (batch_norm) -> activation -> (dropout)
#     batch norm and dropout are optional
#     """
#     def __init__(self, in_channels, out_channels,depth, kernel_size, dim = 3,
#             stride = 1, padding = 0, dilation =1, groups = 1, bias = True, 
#             batch_norm = True, acti_func = None):
#         super(ConvolutionLayerSNEDepth, self).__init__()
#         self.n_in_chns  = in_channels
#         self.n_out_chns = out_channels
#         self.batch_norm = batch_norm
#         self.acti_func  = acti_func
#         self.sne_rnn_block = SE_BlockV2(channel = out_channels,depth = depth)

#         assert(dim == 2 or dim == 3)
#         if(dim == 2):
#             self.conv = nn.Conv2d(in_channels, out_channels,
#                 kernel_size, stride, padding, dilation, groups, bias)
#             if(self.batch_norm):
#                 self.bn = nn.modules.BatchNorm2d(out_channels)
#         else:        
#             self.conv = nn.Conv3d(in_channels, out_channels,
#                 kernel_size, stride, padding, dilation, groups, bias)
#             if(self.batch_norm):
#                 self.bn = nn.modules.BatchNorm3d(out_channels)
#                 # self.bn = nn.modules.InstanceNorm3d(out_channels)

#     def forward(self, x):
#         f = self.conv(x)
#         f = self.sne_rnn_block(f)
#         if(self.batch_norm):
#             f = self.bn(f)
#         if(self.acti_func is not None):
#             f = self.acti_func(f)
#         return f

class SE_BlockV3(nn.Module):
    def __init__(self, channel,r=4):
        super(SE_BlockV3,self).__init__()
        self.squeeze_channel = nn.AdaptiveAvgPool3d((1,1,1) )

        self.excitation_channel = nn.Sequential(
            nn.Linear(channel, channel//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//r, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c,d, _, _ = x.shape

        y_channel_squeezed = self.squeeze_channel(x).view(bs,c)
        y_channel_excited = self.excitation_channel(y_channel_squeezed).view(bs,c,1,1,1)
         
        return x * y_channel_excited.expand_as(x)
class ConvolutionLayerSNEAll(nn.Module):
    """
    A compose layer with the following components:
    convolution -> (batch_norm) -> activation -> (dropout)
    batch norm and dropout are optional
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim = 3,
            stride = 1, padding = 0, dilation =1, groups = 1, bias = False, 
            batch_norm = True, acti_func = None):
        super(ConvolutionLayerSNEAll, self).__init__()
        self.n_in_chns  = in_channels
        self.n_out_chns = out_channels
        self.batch_norm = batch_norm
        self.acti_func  = acti_func
        self.sne_all = SE_BlockV3(channel = out_channels)

        assert(dim == 2 or dim == 3)
        if(dim == 2):
            self.conv = nn.Conv2d(in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias)
            if(self.batch_norm):
                self.bn = nn.modules.BatchNorm2d(out_channels)
        else:        
            self.conv = nn.Conv3d(in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias)
            if(self.batch_norm):
                self.bn = nn.modules.BatchNorm3d(out_channels)
                # self.bn = nn.modules.InstanceNorm3d(out_channels)

    def forward(self, x):
        f = self.conv(x)
        if(self.batch_norm):
            f = self.bn(f)
        f = self.sne_all(f)
        if(self.acti_func is not None):
            f = self.acti_func(f)
        return f

class ConvolutionLayerSNEAllV2(nn.Module):
    """
    A compose layer with the following components:
    convolution -> (batch_norm) -> activation -> (dropout)
    batch norm and dropout are optional
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim = 3,
            stride = 1, padding = 0, dilation =1, groups = 1, bias = False, 
            batch_norm = True, acti_func = None):
        super(ConvolutionLayerSNEAllV2, self).__init__()
        self.n_in_chns  = in_channels
        self.n_out_chns = out_channels
        self.batch_norm = batch_norm
        self.acti_func  = acti_func
        self.sne_all = SE_BlockV3(channel = out_channels)

        assert(dim == 2 or dim == 3)
        if(dim == 2):
            self.conv1 = nn.Conv2d(in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias)
            if(self.batch_norm):
                self.bn1 = nn.modules.BatchNorm2d(out_channels)
        else:        
            self.conv = nn.Conv3d(in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias)
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias)
            if(self.batch_norm):
                self.bn = nn.modules.BatchNorm3d(out_channels)
                self.bn2 = nn.modules.BatchNorm3d(out_channels)
                # self.bn = nn.modules.InstanceNorm3d(out_channels)
        self.downsample = nn.Sequential(*[nn.Conv3d(in_channels, out_channels,kernel_size=1, stride=1, bias=bias), nn.modules.BatchNorm3d(out_channels)])

    def forward(self, x):
        identity = x
        f = self.conv(x)
        f = self.bn(f)
        f = self.acti_func(f)

        f = self.conv2(f)
        f = self.bn2(f)
        f = self.sne_all(f)
        identity = self.downsample(identity)
        f += identity
        f = self.acti_func(f)
        return f


class UNetBlockSNEAll(nn.Module):
    def __init__(self, in_channels, out_channels, 
            kernel_size, paddding, acti_func, acti_func_param):
        super(UNetBlockSNEAll, self).__init__()
        
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func

        # self.conv1 = ConvolutionLayer(in_channels,  out_channels, kernel_size = kernel_size, 
        #         padding = paddding, acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv1 = ConvolutionLayerSNEAll(in_channels, out_channels, kernel_size = kernel_size, 
                padding = paddding, acti_func=get_acti_func(acti_func, acti_func_param))
        self.conv2 = ConvolutionLayerSNEAll(out_channels, out_channels, kernel_size = kernel_size, 
                padding = paddding, acti_func=get_acti_func(acti_func, acti_func_param))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNetBlockSNEAllV2(nn.Module):
    def __init__(self, in_channels, out_channels, 
            kernel_size, paddding, acti_func, acti_func_param):
        super(UNetBlockSNEAllV2, self).__init__()
        
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func

        self.conv = ConvolutionLayerSNEAllV2(in_channels, out_channels, kernel_size = kernel_size, 
                padding = paddding, acti_func=get_acti_func(acti_func, acti_func_param))

    def forward(self, x):
        out = self.conv(x)
        return out

# class UNetBlockSNEDepth(nn.Module):
#     def __init__(self, in_channels, out_channels, 
#             kernel_size, paddding, acti_func, acti_func_param,depth=24):
#         super(UNetBlockSNEDepth, self).__init__()
        
#         self.in_chns   = in_channels
#         self.out_chns  = out_channels
#         self.acti_func = acti_func

#         self.conv1 = ConvolutionLayerSNEDepth(in_channels,  out_channels,depth, 
#                                               kernel_size = kernel_size, padding = paddding, 
#                                             acti_func=get_acti_func(acti_func, acti_func_param))
#         self.conv2 = ConvolutionLayerSNEDepth(out_channels, out_channels,depth, 
#                                               kernel_size = kernel_size, padding = paddding, 
#                                               acti_func=get_acti_func(acti_func, acti_func_param))

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x


class SE_BlockDepth(nn.Module):
    def __init__(self, channel,r=4):
        super(SE_BlockDepth,self).__init__()
        self.squeeze_channel = nn.AdaptiveAvgPool3d((1,1,1) )

        self.excitation_channel = nn.Sequential(
            nn.Linear(channel, channel//r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//r, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c,d, _, _ = x.shape

        y_channel_squeezed = self.squeeze_channel(x).view(bs,c)
        y_channel_excited = self.excitation_channel(y_channel_squeezed).view(bs,c,1,1,1)
         
        return x * y_channel_excited.expand_as(x)

class ConvolutionLayerSNEAllV3(nn.Module):
    """
    A compose layer with the following components:
    convolution -> (batch_norm) -> activation -> (dropout)
    batch norm and dropout are optional
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim = 3,
            stride = 1, padding = 0, dilation =1, groups = 1, bias = False, 
            batch_norm = True, acti_func = None):
        super(ConvolutionLayerSNEAllV3, self).__init__()
        self.n_in_chns  = in_channels
        self.n_out_chns = out_channels
        self.batch_norm = batch_norm
        self.acti_func  = acti_func
        self.sne_all = SE_BlockV3(channel = out_channels)

        assert(dim == 2 or dim == 3)
        if(dim == 2):
            self.conv1 = nn.Conv2d(in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias)
            if(self.batch_norm):
                self.bn1 = nn.modules.BatchNorm2d(out_channels)
        else:        
            self.conv = nn.Conv3d(in_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias)
            self.conv2 = nn.Conv3d(out_channels, out_channels,
                kernel_size, stride, padding, dilation, groups, bias)
            if(self.batch_norm):
                self.bn = nn.modules.BatchNorm3d(out_channels)
                self.bn2 = nn.modules.BatchNorm3d(out_channels)
        self.downsample = nn.Sequential(*[nn.Conv3d(in_channels, out_channels,kernel_size=1, stride=1, bias=bias), nn.modules.BatchNorm3d(out_channels)])

    def forward(self, x):
        identity = x
        f = self.conv(x)
        f = self.bn(f)
        f = self.acti_func(f)

        f = self.conv2(f)
        f = self.bn2(f)
        f = self.sne_all(f)
        identity = self.downsample(identity)
        f += identity
        f = self.acti_func(f)
        return f

class ProjectExciteLayer(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                                    squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                                    squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)])

        # Excitation:
        final_squeeze_tensor = self.sigmoid(self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor

class ProjectExciteLayerV2(nn.Module):
    """
        Project & Excite Module, specifically designed for 3D inputs
        *quote*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ProjectExciteLayerV2, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.relu = nn.ReLU()
        self.conv_c = nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1)
        self.conv_cT = nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()

        # Project:
        # Average along channels and different axes
        squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))

        squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))

        squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))

        # tile tensors to original size and add:
        final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),
                                    squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),
                                    squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)])
        # spatial_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1))


        # final_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1),squeeze_tensor_d.view(batch_size, num_channels, D, 1, 1)])

        # Excitation:
        final_squeeze_tensor = self.sigmoid(self.conv_cT(self.relu(self.conv_c(final_squeeze_tensor))))
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor


class UNet2D5(nn.Module):
    def __init__(self, params):
        super(UNet2D5, self).__init__()
        self.params = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.n_class   = self.params['class_num']
        self.acti_func = self.params['acti_func']
        assert(len(self.ft_chns) == 5)

        self.block1 = UNetBlock(self.in_chns, self.ft_chns[0], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block2 = UNetBlock(self.ft_chns[0], self.ft_chns[1], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block3 = UNetBlock(self.ft_chns[1], self.ft_chns[2], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block4 = UNetBlock(self.ft_chns[2], self.ft_chns[3], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block5 = UNetBlock(self.ft_chns[3], self.ft_chns[4], 
            (3, 3, 3), (1, 1, 1), self.acti_func, self.params)

        self.block6 = UNetBlock(self.ft_chns[3] * 2, self.ft_chns[3], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block7 = UNetBlock(self.ft_chns[2] * 2, self.ft_chns[2], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block8 = UNetBlock(self.ft_chns[1] * 2, self.ft_chns[1], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.block9 = UNetBlock(self.ft_chns[0] * 2, self.ft_chns[0], 
            (1, 3, 3), (0, 1, 1), self.acti_func, self.params)

        self.down1 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))
        self.down2 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))
        self.down3 = nn.MaxPool3d(kernel_size = (1, 2, 2), stride = (1, 2, 2))
        # self.down3 = nn.MaxPool3d(kernel_size = 2)
        self.down4 = nn.MaxPool3d(kernel_size = 2)

        self.up1 = DeconvolutionLayer(self.ft_chns[4], self.ft_chns[3], kernel_size = 2,
            stride = 2, acti_func = get_acti_func(self.acti_func, self.params))

        self.up2 = DeconvolutionLayer(self.ft_chns[3], self.ft_chns[2], kernel_size = (1, 2, 2),
            stride = (1, 2, 2), acti_func = get_acti_func(self.acti_func, self.params))

        # self.up2 = DeconvolutionLayer(self.ft_chns[3], self.ft_chns[2], kernel_size = 2,
        #     stride = 2, acti_func = get_acti_func(self.acti_func, self.params))

        self.up3 = DeconvolutionLayer(self.ft_chns[2], self.ft_chns[1], kernel_size = (1, 2, 2),
            stride = (1, 2, 2), acti_func = get_acti_func(self.acti_func, self.params))
        self.up4 = DeconvolutionLayer(self.ft_chns[1], self.ft_chns[0], kernel_size = (1, 2, 2),
            stride = (1, 2 ,2), acti_func = get_acti_func(self.acti_func, self.params))

        self.conv = nn.Conv3d(self.ft_chns[0], self.n_class, 
            kernel_size = (1, 3, 3), padding = (0, 1, 1))

        # self.pe1 = ProjectExciteLayer(self.ft_chns[0])
        # self.pe2 = ProjectExciteLayer(self.ft_chns[1])
        # self.pe3 = ProjectExciteLayer(self.ft_chns[2])
        # self.pe4 = ProjectExciteLayer(self.ft_chns[3])
        # self.pe5_v2 = ProjectExciteLayer(self.ft_chns[4])
        # self.pe6 = ProjectExciteLayer(self.ft_chns[2] * 2)
        # self.pe7 = ProjectExciteLayer(self.ft_chns[1] * 2)
        # self.pe8 = ProjectExciteLayer(self.ft_chns[0] * 2)
        # self.pe9 = ProjectExciteLayer(self.ft_chns[0])


    def forward(self, x):
        f1 = self.block1(x);  d1 = self.down1(f1)
        # d1 = self.pe1(d1)
        # print(f1.shape,d1.shape,'d1')
        f2 = self.block2(d1); d2 = self.down2(f2)
        # d2 = self.pe2(d2)
        # print(f2.shape,d2.shape,'d2')
        f3 = self.block3(d2); d3 = self.down3(f3)
        # d3 = self.pe3(d3)
        # print(f3.shape,d3.shape,'d3')
        f4 = self.block4(d3); d4 = self.down4(f4)
        # d4 = self.pe4(d4)
        # print(f4.shape,d4.shape,'d4')
        f5 = self.block5(d4)
        # f5 = self.pe5_v2(f5)
        # print(f5.shape,'f5')

        f5up  = self.up1(f5)
        # print(f5up.shape,'f5up')
        f4cat = torch.cat((f4, f5up), dim = 1)

        # f4cat = self.pe5(f4cat)

        # print(f4cat.shape,'f4cat')
        f6    = self.block6(f4cat)
        # print(f6.shape,'f6')
        f6up  = self.up2(f6)
        # print(f6up.shape,'f6up')
        f3cat = torch.cat((f3, f6up), dim = 1)

        # f3cat = self.pe6(f3cat)

        f7    = self.block7(f3cat)
        f7up  = self.up3(f7)
        f2cat = torch.cat((f2, f7up), dim = 1)

        # f2cat = self.pe7(f2cat)

        f8    = self.block8(f2cat)
        f8up  = self.up4(f8)
        f1cat = torch.cat((f1, f8up), dim = 1)

        # f1cat = self.pe8(f1cat)

        f9    = self.block9(f1cat)
        # f9 = self.pe9(f9)
        output = self.conv(f9)

        return output
    
    def apply_eval_encoder(self):
        print("Applying Eval On Encoder Blocks")
        self.block1.eval()
        self.block2.eval()
        self.block3.eval()
        self.block4.eval()
        self.block5.eval()        

    def freeze_encoder(self,encoder_dict_keys):
        for name,param in self.named_parameters():
            if name in encoder_dict_keys:
                param.requires_grad = False        
        # self.apply_eval_encoder()


        self.encoder_keys = encoder_dict_keys

    def unfreeze_encoder(self):
        for name,param in self.named_parameters(): 
            if name in self.encoder_keys:
                param.requires_grad = True

        # self.block1.train()
        # self.block2.train()
        # self.block3.train()
        # self.block4.train()
        # self.block5.train()




if __name__ == "__main__":
    params = {'in_chns':4,
              'feature_chns':[2, 8, 32, 48, 64],
              'class_num': 2,
              'acti_func': 'leakyrelu',
              'leakyrelu_alpha': 0.01}
    Net = UNet2D5(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 32, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    y = y.detach().numpy()
    print(y.shape)

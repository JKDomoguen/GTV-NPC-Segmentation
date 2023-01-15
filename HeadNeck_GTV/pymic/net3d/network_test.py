
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

def get_acti_func(acti_func, params={"leakyrelu_negative_slope":0.01}):
    acti_func = acti_func.lower()
    if(acti_func == 'relu'):
        inplace = params.get('relu_inplace', False)
        return nn.ReLU(inplace)

    elif(acti_func == 'leakyrelu'):
        slope   = params.get('leakyrelu_negative_slope', 1e-2)
        inplace = params.get('leakyrelu_inplace', False)
        return nn.LeakyReLU(slope, inplace)

    elif(acti_func == 'prelu'):
        num_params = params.get('prelu_num_parameters', 1)
        init_value = params.get('prelu_init', 0.25)
        return nn.PReLU(num_params, init_value)

    elif(acti_func == 'rrelu'):
        lower   = params.get('rrelu_lower', 1.0 /8)
        upper   = params.get('rrelu_upper', 1.0 /3)
        inplace = params.get('rrelu_inplace', False)
        return nn.RReLU(lower, upper, inplace)

    elif(acti_func == 'elu'):
        alpha   = params.get('elu_alpha', 1.0)
        inplace = params.get('elu_inplace', False)
        return nn.ELU(alpha, inplace)

    elif(acti_func == 'celu'):
        alpha   = params.get('celu_alpha', 1.0)
        inplace = params.get('celu_inplace', False)
        return nn.CELU(alpha, inplace)

    elif(acti_func == 'selu'):
        inplace = params.get('selu_inplace', False)
        return nn.SELU(inplace)

    elif(acti_func == 'glu'):
        dim = params.get('glu_dim', -1)
        return nn.GLU(dim)

    elif(acti_func == 'sigmoid'):
        return nn.Sigmoid()

    elif(acti_func == 'logsigmoid'):
        return nn.LogSigmoid()

    elif(acti_func == 'tanh'):
        return nn.Tanh()

    elif(acti_func == 'hardtanh'):
        min_val = params.get('hardtanh_min_val', -1.0)
        max_val = params.get('hardtanh_max_val',  1.0)
        inplace = params.get('hardtanh_inplace', False)
        return nn.Hardtanh(min_val, max_val, inplace)
    
    elif(acti_func == 'softplus'):
        beta      = params.get('softplus_beta', 1.0)
        threshold = params.get('softplus_threshold', 20)
        return nn.Softplus(beta, threshold)
    
    elif(acti_func == 'softshrink'):
        lambd = params.get('softshrink_lambda', 0.5)
        return nn.Softshrink(lambd)
    
    elif(acti_func == 'softsign'):
        return nn.Softsign()
    
    else:
        raise ValueError("Not implemented: {0:}".format(acti_func))


class ConvolutionLayer(nn.Module):
    """
    A compose layer with the following components:
    convolution -> (batch_norm) -> activation -> (dropout)
    batch norm and dropout are optional
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim = 3,
            stride = 1, padding = 0, dilation =1, groups = 1, bias = True, 
            batch_norm = True, acti_func = None):
        super(ConvolutionLayer, self).__init__()
        self.n_in_chns  = in_channels
        self.n_out_chns = out_channels
        self.batch_norm = batch_norm
        self.acti_func  = acti_func

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
        if(self.acti_func is not None):
            f = self.acti_func(f)
        return f

class DeconvolutionLayer(nn.Module):
    """
    A compose layer with the following components:
    deconvolution -> (batch_norm) -> activation -> (dropout)
    batch norm and dropout are optional
    """
    def __init__(self, in_channels, out_channels, kernel_size, 
            dim = 3, stride = 1, padding = 0, output_padding = 0, 
            dilation =1, groups = 1, bias = True, 
            batch_norm = True, acti_func = None):
        super(DeconvolutionLayer, self).__init__()
        self.n_in_chns  = in_channels
        self.n_out_chns = out_channels
        self.batch_norm = batch_norm
        self.acti_func  = acti_func
        
        assert(dim == 2 or dim == 3)
        if(dim == 2):
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                kernel_size, stride, padding, output_padding,
                groups, bias, dilation)
            if(self.batch_norm):
                self.bn = nn.modules.BatchNorm2d(out_channels)
        else:
            self.conv = nn.ConvTranspose3d(in_channels, out_channels,
                kernel_size, stride, padding, output_padding,
                groups, bias, dilation)
            if(self.batch_norm):
                self.bn = nn.modules.BatchNorm3d(out_channels)

    def forward(self, x):
        f = self.conv(x)
        if(self.batch_norm):
            f = self.bn(f)
        if(self.acti_func is not None):
            f = self.acti_func(f)
        return f

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
            kernel_size, paddding, acti_func):
        super(UNetBlock, self).__init__()
        
        self.in_chns   = in_channels
        self.out_chns  = out_channels
        self.acti_func = acti_func

        self.conv1 = ConvolutionLayer(in_channels,  out_channels, kernel_size = kernel_size, 
                padding = paddding, acti_func=get_acti_func(acti_func))
        self.conv2 = ConvolutionLayer(out_channels, out_channels, kernel_size = kernel_size, 
                padding = paddding, acti_func=get_acti_func(acti_func))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


if __name__ == "__main__":
    ft_chns = [16, 32, 64, 128, 256]
    n_class = 2
    acti_func = "leakyrelu"
    in_chns = 1

    spatial_block = UNetBlock(in_chns, in_chns, (1, 3, 3), (0, 1, 1), acti_func)
    depth_block = UNetBlock(in_chns, in_chns, (3,1,1), (1, 0, 0), acti_func)

    input_tensor = torch.randn(16,1,16,64,64)
    batch_size, num_channels, D, H, W = input_tensor.size()
    squeeze_tensor_w = F.adaptive_avg_pool3d(input_tensor, (1, 1, W))
    squeeze_tensor_h = F.adaptive_avg_pool3d(input_tensor, (1, H, 1))
    squeeze_tensor_d = F.adaptive_avg_pool3d(input_tensor, (D, 1, 1))
    
    spatial_squeeze_tensor = sum([squeeze_tensor_w.view(batch_size, num_channels, 1, 1, W),squeeze_tensor_h.view(batch_size, num_channels, 1, H, 1)])
    output = spatial_squeeze_tensor*squeeze_tensor_d
    # print(output.shape,spatial_squeeze_tensor.shape,squeeze_tensor_d.shape)
    block1_output = spatial_block(spatial_squeeze_tensor)
    depth_output = depth_block(squeeze_tensor_d)
    print(block1_output.shape,spatial_squeeze_tensor.shape)
    print(depth_output.shape,squeeze_tensor_d.shape)
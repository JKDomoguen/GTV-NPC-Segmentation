
from __future__ import print_function

import torch
import torch.nn as nn

from pymic.self_supervised_tasks.networks.encoder_networks import UNet2D5Encoder
from pymic.self_supervised_tasks.networks.encoder_networks import UNet3DEncoder

from pymic.self_supervised_tasks.networks.encoder_networks import UNet2D5Encoder
from pymic.self_supervised_tasks.networks.encoder_networks import UNet3DEncoder

from pymic.self_supervised_tasks.networks.dense_networks import FullyConnected
from pymic.self_supervised_tasks.networks.dense_networks import SimpleMulticlass
from pymic.self_supervised_tasks.networks.dense_networks import FullyConnectedBig

class RotationModel(nn.Module):
    def __init__(self,params):
        super(RotationModel,self).__init__()
        input_size = params['SSL']['input_size']
        hidden_size = params['SSL']['hidden_size']
        include_top = params['SSL']['include_top']
        num_class = params['SSL']['num_class']
        self.batch_size_val = 4
        self.batch_size = params['training']['batch_size']
        

        self.params = params
        if self.params['SSL']['encoder'] == 'Unet3D':
            self._encoder = UNet3DEncoder(self.params['network']) 
        elif self.params['SSL']['encoder'] == 'Unet2D5':
            self._encoder = UNet2D5Encoder(self.params['network'],down3=False) 
        if self.params['SSL']['submodel'] == 'simple':
            self._submodel = FullyConnected(input_size=input_size,hidden_size=hidden_size,include_top=include_top)
        elif self.params['SSL']['submodel'] == 'multi_class':
            self._submodel = SimpleMulticlass(input_size=input_size,hidden_size=hidden_size,include_top=include_top)
        elif self.params['SSL']['submodel'] == 'fullybig':
            self._submodel = FullyConnectedBig(input_size=input_size,hidden_size=hidden_size,include_top=include_top) 
        
        final_layer = []
        final_layer.append(nn.Linear(hidden_size//2,num_class))
        # final_layer.append(nn.Softmax(dim=1))
        self._final_layer = nn.Sequential(*final_layer)
        self.down = nn.AdaptiveAvgPool3d((None,1,1))
        
        
    def forward(self,x):
        out = self._encoder(x) 
        out = self.down(out)
        out = out.view(self.batch_size,-1)
        out = self._submodel(out)
        return self._final_layer(out)

    def forward_val(self,x):
        out = self._encoder(x)
        out = self.down(out)
        out = out.view(self.batch_size_val,-1)
        out = self._submodel(out)
        return self._final_layer(out)

# class RPL_Trainer:
#     def __init__(
#         self,
#         params,
#         data_dim=384,
#         patches_per_side=3,
#         patch_jitter=0,
#         ):
#         self.patch_jitter = patch_jitter
#         self.patches_per_side = patches_per_side
#         self.patch_dim = (data_dim // patches_per_side) - patch_jitter

#         self.patch_shape = (self.patch_dim,) + self.patch_shape
#         self.patch_count = self.patches_per_side ** 3

#         self.images_shape = (2,) + self.patch_shape
#         self.class_count = self.patch_count - 1
#         self._rpl_model = RelativePatchLocationModel(params,self.patch_shape,self.class_count)
        
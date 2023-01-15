
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

class MatchingModel(nn.Module):
    def __init__(self,params):
        super(MatchingModel,self).__init__()
        input_size = params['SSL']['input_size']
        hidden_size = params['SSL']['hidden_size']
        include_top = params['SSL']['include_top']
        self.batch_size_val = params['training']['batch_size']
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
        
        self.down = nn.AdaptiveAvgPool3d((None,1,1))
        
        
    def forward(self,x):
        out = self._encoder(x) 
        out = self.down(out).view(out.shape[0],-1)
        out = self._submodel(out)
        out = out.view(self.batch_size,3,-1)
        return out

    def forward_val(self,x):
        out = self._encoder(x)
        out = self.down(out).view(out.shape[0],-1)
        out = self._submodel(out)
        out = out.view(self.batch_size_val,3,-1)
        # print(out.shape,'submodel out')
        return out
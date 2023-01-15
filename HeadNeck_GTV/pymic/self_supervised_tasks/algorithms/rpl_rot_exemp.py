
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

class RPL_ROT_EXEMP_MODEL(nn.Module):
    def __init__(self,params):
        super(RPL_ROT_EXEMP_MODEL,self).__init__()
        input_size = params['SSL']['input_size']
        hidden_size = params['SSL']['hidden_size']
        input_size_rot = 6144
        input_size_exemp = 12288
        num_class_rpl = 26
        num_class_rot = 10
        include_top = params['SSL']['include_top']
        self.batch_size = params['training']['batch_size']

        self.params = params
        if self.params['SSL']['encoder'] == 'Unet3D':
            self._encoder = UNet3DEncoder(self.params['network']) 
        elif self.params['SSL']['encoder'] == 'Unet2D5':
            self._encoder = UNet2D5Encoder(self.params['network']) 

        if self.params['SSL']['submodel'] == 'simple':
            self._submodel = FullyConnected(input_size=input_size,hidden_size=hidden_size,include_top=include_top)
        elif self.params['SSL']['submodel'] == 'multi_class':
            self._submodel = SimpleMulticlass(input_size=input_size,hidden_size=hidden_size,include_top=include_top)
        elif self.params['SSL']['submodel'] == 'fullybig':
            self._submodel_rpl = FullyConnectedBig(input_size=input_size,hidden_size=hidden_size,include_top=include_top)
            self._submodel_rot = FullyConnectedBig(input_size=input_size_rot,hidden_size=hidden_size,include_top=include_top)  

                   
        final_layer = []
        final_layer.append(nn.Linear(hidden_size//2,num_class_rpl))
        self._final_layer_rpl = nn.Sequential(*final_layer)

        final_layer = []
        final_layer.append(nn.Linear(hidden_size//2,num_class_rot))
        self._final_layer_rot = nn.Sequential(*final_layer)

        self.down = nn.AdaptiveAvgPool3d((None,1,1))
        
    def forward_rpl(self,x):
        out = self._encoder(x)
        out = self.down(out).view(self.batch_size,-1)
        out = self._submodel_rpl(out)
        out = self._final_layer_rpl(out)
        return out

    def forward_rot(self,x):
        out = self._encoder(x)
        out = self.down(out).view(self.batch_size,-1)
        out = self._submodel_rot(out)
        out = self._final_layer_rot(out)
        return out

    def forward_exemp(self,x):
        out = self._encoder(x)
        out = self.down(out).view(out.shape[0],-1)
        out = self._submodel_rot(out)
        out = out.view(self.batch_size,2,-1)
        return out

    def forward_val_rpl(self,x):
        out = self._encoder(x)
        # print(out.shape,'output-shape')
        # print(self.down(out).shape,'down-shape')
        out = self.down(out).view(4,-1)
        out = self._submodel_rpl(out)
        out = self._final_layer_rpl(out)
        return out

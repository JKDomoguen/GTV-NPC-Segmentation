# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import os

from pymic.net3d.unet2d5 import UNet2D5
from pymic.net3d.unet3d import UNet3D
from pymic.net3d.unet3d_genesis import UNet3DGenesis
from pymic.net3d.baseunet2d5_att_pe import Baseunet2d5_att_pe
from pymic.net3d.baseunet2d5_pe import UNet2D5_PE
from pymic.net3d.vnet import VNet



def get_network(params):
    net_type = params['net_type']
    if(net_type == 'UNet2D5'):
        model =  UNet2D5(params)
    elif(net_type == 'Baseunet2d5_att_pe'):
        model = Baseunet2d5_att_pe(params)
    elif(net_type == 'VNet'):
        model = VNet()
    elif(net_type == 'UNet2D5_PE'):
        model = UNet2D5_PE(params)    
    elif(net_type == 'UNet3D'):
        model = UNet3D(params)
    elif(net_type == 'UNet3DGenesis'):
        model = UNet3DGenesis(n_class=params['class_num'])
    else:
        raise ValueError("undefined network {0:}".format(net_type))

    if params['use_pretrain'] and os.path.isfile(params['pretrained_model_path']):
        print('\n\n\n********************************')
        print('Using a pre-trained Model')
        print('********************************\n')
        checkpoint = torch.load(params['pretrained_model_path'])

        if net_type == 'UNet3DGenesis':
            state_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            unParalled_state_dict = {}
            for key in state_dict.keys():
                # print('model',key)
                if 'out_tr' in key.lower():
                    continue
                unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            state_dict = unParalled_state_dict
            model_dict.update(state_dict)
            state_dict = model_dict
        else:
            state_dict = checkpoint['model_state_dict']
            model_dict = model.state_dict()
            for state_key,state_value in state_dict.items():
                if state_key in model_dict:
                    if model_dict[state_key].shape == state_value.shape:
                        model_dict[state_key] = state_value
                        print(f'Using a Pretrain Layer:{state_key} with Tensor Shape:{state_value.shape}')
        model.load_state_dict(model_dict)
        
    print('Network-Type: ',net_type)
    return model

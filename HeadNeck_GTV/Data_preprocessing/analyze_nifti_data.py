import os
import sys
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


from pymic.io.nifty_dataset import NiftyDataset
from pymic.util.parse_config import parse_config
from pymic.io.transform3d import get_transform

from pymic.self_supervised_tasks.dataset.nifi_dataset_ssl import NiftyDataset_SSL

from pymic.self_supervised_tasks.networks.encoder_networks import UNet2D5Encoder
from pymic.self_supervised_tasks.networks.dense_networks import FullyConnectedBig

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

def analyze_nifi_data(data_path,train_csv_path,val_csv_path):

    train_dataset = NiftyDataset(root_dir=data_path,
                                 csv_file=train_csv_path,
                                 modal_num=1,
                                 with_label=True,
                                 )
    # valid_dataset = NiftyDataset(root_dir=data_path,
    #                              csv_file=val_csv_path,
    #                              modal_num=1,
    #                              with_label=True,
    #                              )
    
    train_loader = torch.utils.data.DataLoader(train_dataset,pin_memory=True,
                batch_size = 32, shuffle=True, num_workers=2)

    # valid_loader = torch.utils.data.DataLoader(valid_dataset,pin_memory=True,
    #             batch_size = 1, shuffle=False, num_workers=2)
    failed_iter = 0
    print('dafaq')
    for idx,data in enumerate(train_loader):
        inputs, labels = data['image'], data['label']
        print(idx,' ',inputs.shape,np.unique(labels))
        if len(np.unique(labels)) != 2:
            failed_iter += 1
            print("Failed Iter:{}".format(data['names']))

    
    print("Number of Failed Iter:{}".format(failed_iter))

def analyze_nifi_data_ssl(data_path,data_csv_path):

    train_dataset = NiftyDataset_SSL(root_dir=data_path,
                                 csv_file=data_csv_path,
                                 )
    BATCH_SIZE = 8
    NUM_CLASS = 2
    data_loader = torch.utils.data.DataLoader(train_dataset,pin_memory=True,
                batch_size = BATCH_SIZE, shuffle=True, num_workers=4)
    params = {'in_chns':1,'feature_chns':[16, 32, 64, 128, 256],'class_num':2,'acti_func':'leakyrelu','dropout':False}
    model = UNet2D5Encoder(params)
    fcn = FullyConnectedBig(include_top=False)
    # model = UNet3DEncoder(params)
    model.cuda()
    fcn.cuda()
    model.eval()

    for idx,data in enumerate(data_loader):
        with torch.no_grad():            
            inputs, labels = data['image'].cuda(), data['label'].cuda()
            inputs = inputs.view(NUM_CLASS*BATCH_SIZE,1,18,47,47)
            encoded_out = model(inputs).view(BATCH_SIZE,-1)
            out_dense = fcn(encoded_out)
            print(encoded_out.shape,out_dense.shape)

if __name__ == '__main__':
    # data_path = '../../Dataset_Rad2/gtv_train_processed_sliced2/uncut_scale'
    # data_path2 = '../../Dataset_Rad2/gtv_train_processed_sliced/large_scale'
    # data_path3 = '../../Dataset_Rad2/gtv_train_processed_sliced/middle_scale'
    # train_csv_path = 'config/gtv_config_sliced2/gtv_train.csv'
    # val_csv_path = 'config/gtv_config_sliced2/gtv_valid.csv'

    # data_path = '../../Dataset_Rad2/gtv_train_processed_sliced/middle_scale'
    # train_csv_path = 'config/gtv_sliced_config/gtv_train_full.csv'

    data_path = "../../Dataset_Rad2/gtv_normal_unlabelled/middle_scale"
    data_csv_path = 'config/gtv_normal_sliced_config/gtv_train_full_middle.csv'

    # data_path = '/home/jansendomoguen/jansen/code_rad_cvmig02/Dataset_Rad2/gtv_train_processed/small_scale'
    # data_csv_path = 'config/gtv_pure_config/gtv_train_full.csv'
    analyze_nifi_data_ssl(data_path,data_csv_path)
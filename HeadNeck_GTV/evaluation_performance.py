# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import sys
import time
import math

import scipy
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import csv
import copy
import nibabel
import statistics

torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# file_dir = os.path.dirname(__file__)
# sys.path.append(file_dir)
# sys.path.append("../../pymic")
# sys.path.append("../../pymic/")
from datetime import datetime
from tensorboardX import SummaryWriter
from pymic.io.nifty_dataset import NiftyDataset
from pymic.io.transform3d import get_transform
from pymic.train_infer.net_factory import get_network
from pymic.train_infer.loss import *
from pymic.train_infer.get_optimizer import get_optimiser
from pymic.util.image_process import convert_label
from pymic.util.parse_config import parse_config
from pymic.util.evaluation import ppv_func, sensitivity, specificity
from pymic.util.evaluation import binary_assd, binary_iou, binary_hausdorff, binary_relative_volume_error, binary_dice
from pymic.net3d.unet_3D_dv_semi import unet_3D_dv_semi
import matplotlib.pyplot as plt
from PIL import Image

def load_origin_nifty_volume_as_array(filename):
    """
    load nifty image into numpy array, and transpose it based on the [z,y,x] axis order
    The output array shape is like [Depth, Height, Width]
    inputs:
        filename: the input file name, should be *.nii or *.nii.gz
    outputs:
        data: a numpy data array
        zoomfactor:
    """
    img = nibabel.load(filename)
    pixelspacing = img.header.get_zooms()
    zoomfactor = list(pixelspacing)
    zoomfactor.reverse()
    # data = img.get_data()
    data = np.asanyarray(img.dataobj)
    data = data.transpose(2, 1, 0)
#     print(data.shape)

    return data, zoomfactor


torch.backends.cudnn.benchmark = True

def running_mean(x, N=50):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class TrainInferAgent():
    def __init__(self, config, stage = 'train',exp='debug'):
        self.config = config
        self.stage  = stage

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
        folder_name = f"{stage}__{dt_string}"
        result_dir = self.config['testing']['output_dir']
        self.output_dir = os.path.join(result_dir,folder_name)
        self.exp = exp
        if exp != 'debug':        
            os.makedirs(self.output_dir,exist_ok=True)


    def __create_dataset(self):
        root_dir_test  = self.config['dataset']['root_dir_test_uncut']
        test_csv  = self.config['dataset'].get('test_csv', None)
        modal_num = self.config['dataset']['modal_num']
        
        transform_names = self.config['dataset']['test_transform']
        self.transform_list = [get_transform(name, self.config['dataset']) \
            for name in transform_names if name != 'RegionSwop']

        test_dataset = NiftyDataset(root_dir=root_dir_test,
                            csv_file  = test_csv,
                            modal_num = modal_num,
                            with_label= True,
                            transform = transforms.Compose(self.transform_list))
        batch_size = 1
        self.test_loder = torch.utils.data.DataLoader(test_dataset,
            batch_size=batch_size, shuffle=False, num_workers=batch_size)


    def __create_network(self):
        if self.config['network']['net_type'] == 'URPC':
            self.net = unet_3D_dv_semi(n_classes=self.config['network']['class_num'],in_channels=self.config['network']['in_chns'])
            print("\n\n\n***********************************")
            print("Using URPC")
            print("***********************************\n\n\n")
            return
        self.net = get_network(self.config['network'])
        network_dict = self.net.state_dict()
        if self.config['training']['use_pretrain'] and os.path.isfile(self.config['training']['pretrained_model_path']):
            print("\n\n\n***********************************")
            print("Using a Pretrained Network Dict:{}".format(self.config['training']['pretrained_model_path']))
            pretrained_model_path = self.config['training']['pretrained_model_path']
            pretrained_dict = torch.load(pretrained_model_path)['model_state_dict']
            pretrained_net_dict = {k: v for k, v in pretrained_dict.items() if k in network_dict}
            for keys,_ in pretrained_net_dict.items():
                print("Keys that are loaded from Pretrain Networks:{}".format(keys))
            network_dict.update(pretrained_net_dict)
            self.net.load_state_dict(network_dict)
            print("Loading Succesful")
            print("***********************************\n\n\n")


    def __dsc_eval(self,exp):

        if self.exp != 'debug':
            infer_txt = open( os.path.join(self.output_dir,'infer_output_{}.txt'.format(exp)) ,'a')
        else:
            infer_txt = open('infer_output_{}.txt'.format(exp) ,'w')
        device = torch.device(self.config['testing']['device_name'])

        checkpoint_small  = self.config['ensemble']['small']
        checkpoint_middle = self.config['ensemble']['middle']
        checkpoint_large = self.config['ensemble']['large']
        checkpoint_extra_middle = self.config['ensemble']['extra_middle']
        checkpoint_below_large = self.config['ensemble']['below_large']

        ckpt_small = torch.load(checkpoint_small)
        ckpt_middle = torch.load(checkpoint_middle)
        ckpt_large = torch.load(checkpoint_large)
        ckpt_extra_middle = torch.load(checkpoint_extra_middle)
        ckpt_below_large = torch.load(checkpoint_below_large)

        # self.checkpoint = torch.load(checkpoint)
        # self.net.load_state_dict(self.checkpoint['model_state_dict'])
        model_small = copy.deepcopy(self.net)
        model_middle = copy.deepcopy(self.net)
        model_large = copy.deepcopy(self.net)
        model_extra_middle = copy.deepcopy(self.net)
        model_below_large = copy.deepcopy(self.net)

        model_small.load_state_dict(ckpt_small['model_state_dict'])
        model_middle.load_state_dict(ckpt_middle['model_state_dict'])
        model_large.load_state_dict(ckpt_large['model_state_dict'])
        model_extra_middle.load_state_dict(ckpt_extra_middle['model_state_dict'])
        model_below_large.load_state_dict(ckpt_below_large['model_state_dict'])
        
        model_small.to(device)
        model_middle.to(device)
        model_large.to(device)
        model_extra_middle.to(device)
        model_below_large.to(device)

        model_small.eval()
        model_middle.eval()
        model_large.eval()
        model_extra_middle.eval()
        model_below_large.eval()

        print('Starting to test using checkpoint-Small:{}'.format(checkpoint_small))
        print('Starting to test using checkpoint-Small:{}'.format(checkpoint_small),file=infer_txt)

        print('Starting to test using checkpoint-Middle:{}'.format(checkpoint_middle))
        print('Starting to test using checkpoint-Middle:{}'.format(checkpoint_middle),file=infer_txt)

        print('Starting to test using checkpoint-Large:{}'.format(checkpoint_large))
        print('Starting to test using checkpoint-Large:{}'.format(checkpoint_large),file=infer_txt)

        print('Starting to test using checkpoint-Extra-Middle:{}'.format(checkpoint_extra_middle))
        print('Starting to test using checkpoint-Extra-Middle:{}'.format(checkpoint_extra_middle),file=infer_txt)

        print('Starting to test using checkpoint-Below-Large:{}'.format(checkpoint_below_large))
        print('Starting to test using checkpoint-Below-Large:{}'.format(checkpoint_below_large),file=infer_txt)

        class_num  = self.config['network']['class_num']
        SLICE = 16

        start_time = time.time()
        avg_test_dsc_small = []
        avg_test_dsc_middle = []
        avg_test_dsc_large = []
        avg_test_dsc_extra_middle = []
        avg_test_dsc_below_large = []
        avg_test_dsc_ensem_mid = []
        avg_test_dsc_ensem_mid_uncert = []
        avg_test_dsc_out_ave = []

        gtv_variance_np_dir = 'results/feb22_gtv_uncertainty_map'
        os.makedirs(gtv_variance_np_dir,exist_ok=True)
        soft_max_f = nn.Softmax(dim=1)

        orig_data_dir = '../../Dataset_Rad2/nifti_gtv_test'
        with torch.no_grad():
            for iter_test,data in enumerate(self.test_loder):
                images,labels = data['image'],data['label']
                present_classes = torch.unique(labels.flatten()).numpy()
                images, labels = images.to(device), labels.to(device)
                total_gtv_voxels = torch.sum(labels.flatten()).item() 
                names  = data['names']
                nifti_name = os.path.join(orig_data_dir,names[0].replace('data','image'))
                img_nifti, pixel_spacing = load_origin_nifty_volume_as_array(nifti_name)
                img_nifti = torch.from_numpy(np.expand_dims(np.expand_dims(img_nifti,axis=0),axis=0)).to(device)
                # print(img_nifti.shape,images.shape,labels.shape)
                # pixel_spacing = data['spacing']
                print("Processing --> {} with input and mask shape --> {}, {} at Iteration--{}".format(names,images.shape,labels.shape,iter_test))
                print("Processing --> {} with input and mask shape --> {}, {} at Iteration--{}".format(names,images.shape,labels.shape,iter_test),file=infer_txt)
                needed_num_slice = SLICE - int(images.shape[2])%SLICE
                last_inp_slice,last_label_slice = images[0,0,-1,:,:],labels[0,0,-1,:,:]
                inp_slice = last_inp_slice.repeat(1,1,needed_num_slice,1,1)
                label_slice_extra = last_label_slice.repeat(1,1,needed_num_slice,1,1)
                images = torch.cat((images,inp_slice),dim=2)
                labels = torch.cat((labels,label_slice_extra),dim=2)
                img_nifti = torch.cat((img_nifti,inp_slice),dim=2)

                images = torch.split(images,SLICE,2)
                labels = torch.split(labels,SLICE,2)
                img_nifti = torch.split(img_nifti,SLICE,2)

                soft_out_seq_small = []
                soft_out_seq_middle = []
                soft_out_seq_large = []
                soft_out_seq_below_large = []
                soft_out_seq_extra_middle = []
                soft_out_ave_seq = []
                
                soft_label_seq = []
                input_slice_seq = []

                for idx,(inputs_slice,labels_slice) in enumerate(zip(images,labels)):
                    img_nifti_slice = img_nifti[idx]
                    # print(img_nifti_slice.shape)

                    if len(torch.unique(labels_slice))<class_num:
                        continue
                    soft_y = get_soft_label(labels_slice,class_num,device).cpu()
                    total_slice_gtv_voxels = torch.sum(labels_slice.flatten()).cpu().item()
                    ratio_slice_gtv_voxels = round(total_slice_gtv_voxels/total_gtv_voxels,4)


                    #Small
                    output_small = model_small(inputs_slice)
                    outputs_argmax_small = torch.argmax(output_small, dim = 1, keepdim = True)
                    
                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_small = outputs_argmax_small[:,:,:end_slice,:,:]
                        soft_y = soft_y[:,:,:end_slice,:,:]
                        img_nifti_slice = img_nifti_slice[:,:,:end_slice,:,:]


                    soft_out_small  = get_soft_label(outputs_argmax_small, class_num,device).cpu()
                    soft_out_seq_small.append(soft_out_small)
                    soft_label_seq.append(soft_y)                        
                    input_slice_seq.append(img_nifti_slice.cpu())

                    
                    #Middle
                    output_middle = model_middle(inputs_slice)
                    outputs_argmax_middle = torch.argmax(output_middle, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_middle = outputs_argmax_middle[:,:,:end_slice,:,:]

                    soft_out_middle  = get_soft_label(outputs_argmax_middle, class_num,device).cpu()
                    soft_out_seq_middle.append(soft_out_middle)

                    #Large
                    output_large = model_large(inputs_slice)
                    outputs_argmax_large = torch.argmax(output_large, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_large = outputs_argmax_large[:,:,:end_slice,:,:]

                    soft_out_large  = get_soft_label(outputs_argmax_large, class_num,device).cpu()
                    soft_out_seq_large.append(soft_out_large)

                    #Below Large
                    output_below_large = model_below_large(inputs_slice)
                    outputs_argmax_below_large = torch.argmax(output_below_large, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_below_large = outputs_argmax_below_large[:,:,:end_slice,:,:]

                    soft_out_below_large  = get_soft_label(outputs_argmax_below_large, class_num,device).cpu()
                    soft_out_seq_below_large.append(soft_out_below_large)

                    #Extra Middle
                    output_extra_middle = model_extra_middle(inputs_slice)
                    outputs_argmax_extra_middle = torch.argmax(output_extra_middle, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_extra_middle = outputs_argmax_extra_middle[:,:,:end_slice,:,:]

                    soft_out_extra_middle  = get_soft_label(outputs_argmax_extra_middle, class_num,device).cpu()
                    soft_out_seq_extra_middle.append(soft_out_extra_middle)
                    

                    out_small_smax = soft_max_f(output_small.cpu())
                    out_middle_smax = soft_max_f(output_middle.cpu())
                    out_large_smax = soft_max_f(output_large.cpu())
                    out_extra_middle_smax = soft_max_f(output_extra_middle.cpu())
                    out_below_large_smax = soft_max_f(output_below_large.cpu()) 
                    
                    ave_output = (out_middle_smax+out_small_smax+out_large_smax+out_extra_middle_smax+out_below_large_smax)/5
                    ave_output_argmax = torch.argmax(ave_output,dim=1,keepdim=True)
                    soft_out_ave = get_soft_label(ave_output_argmax, class_num,device)
                    soft_out_ave_seq.append(soft_out_ave)


                    del output_extra_middle
                    del outputs_argmax_extra_middle
                    del output_small
                    del outputs_argmax_small
                    del output_middle
                    del outputs_argmax_middle
                    del output_large
                    del outputs_argmax_large
                    del output_below_large
                    del outputs_argmax_below_large


                    dsc_value_iter_small = get_classwise_dice(soft_out_small, soft_y).cpu().numpy()
                    dsc_value_iter_middle = get_classwise_dice(soft_out_middle, soft_y).cpu().numpy()
                    dsc_value_iter_large = get_classwise_dice(soft_out_large, soft_y).cpu().numpy()
                    dsc_value_iter_extra_middle = get_classwise_dice(soft_out_extra_middle, soft_y).cpu().numpy()
                    dsc_value_iter_below_large = get_classwise_dice(soft_out_below_large, soft_y).cpu().numpy()
                    dsc_value_iter_ave = get_classwise_dice(soft_out_ave, soft_y).cpu().numpy()



                    print("Small--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_small,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels))
                    print("Small--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_small,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels),file=infer_txt)

                    print("Middle--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_middle,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels))
                    print("Middle--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_middle,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels),file=infer_txt)

                    print("Large--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_large,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels))
                    print("Large--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_large,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels),file=infer_txt)

                    print("Extra-Middle--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_extra_middle,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels))
                    print("Extra-Middle--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_extra_middle,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels),file=infer_txt)

                    print("Below-Large--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_below_large,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels))
                    print("Below-Large--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_below_large,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels),file=infer_txt)

                    print("Aggregated Output--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_ave,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels))
                    print("Aggregated Output--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_ave,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels),file=infer_txt)

                #Input 
                soft_label_seq = torch.cat(soft_label_seq,dim=2)
                input_slice_seq = torch.cat(input_slice_seq,dim=2)

                #Output
                soft_out_seq_small = torch.cat(soft_out_seq_small,dim=2)
                soft_out_seq_middle = torch.cat(soft_out_seq_middle,dim=2)
                soft_out_seq_large = torch.cat(soft_out_seq_large,dim=2)
                soft_out_seq_extra_middle = torch.cat(soft_out_seq_extra_middle,dim=2)
                soft_out_seq_below_large = torch.cat(soft_out_seq_below_large,dim=2)                
                #Average
                soft_out_ave_seq = torch.cat(soft_out_ave_seq,dim=2)



                summed_soft_out = soft_out_seq_large+soft_out_seq_middle+soft_out_seq_small+soft_out_seq_below_large+soft_out_seq_extra_middle
                soft_out_ensem_zeros = torch.zeros_like(soft_label_seq)
                soft_out_ensem_ones = torch.ones_like(soft_label_seq)
                comb_soft_out = torch.cat([soft_out_seq_large,soft_out_seq_middle,soft_out_seq_small,soft_out_seq_below_large,soft_out_seq_extra_middle],dim=0)


                comb_soft_out_std = torch.std(comb_soft_out,dim=0,unbiased=True,keepdim=True)
                comb_soft_out_mean = torch.mean(comb_soft_out,dim=0,keepdim=True)
                uncertainty_map = comb_soft_out_std/comb_soft_out_mean
                uncertainty_map = torch.nan_to_num(uncertainty_map)
                uncertainty_mask = torch.where(uncertainty_map<=1.0,1.0,0.0)


                soft_out_ensem_mid = torch.where(summed_soft_out>=3,soft_out_ensem_ones,soft_out_ensem_zeros)
                soft_out_ensem_mid_uncert = uncertainty_mask*soft_out_ensem_mid

                # gtv_var_map = comb_soft_out_var[0,1,:,:,:].squeeze().cpu().numpy()
                # soft_label_map = soft_label_seq[0,1,:,:,:].squeeze().cpu().numpy()
                # soft_out_ensem_mid_map = soft_out_ensem_mid[0,1,:,:,:].squeeze().cpu().numpy()
                # input_slice_map = input_slice_seq[0,0,:,:,:].squeeze().cpu().numpy()
                # data_dict = {'gtv_var_map':gtv_var_map,'label_map':soft_label_map,'pred_map':soft_out_ensem_mid_map,'input_map':input_slice_map}
                # names_gtv_np = names[0].split('/')[-1].strip('.nii')
                # print(gtv_var_map.shape,soft_label_map.shape,soft_out_ensem_mid_map.shape,input_slice_map.shape)
                # np.savez(os.path.join(gtv_variance_np_dir,names_gtv_np),data_dict)                



                gtv_dice_small = get_classwise_dice(soft_out_seq_small,soft_label_seq).cpu().numpy()
                gtv_dice_middle = get_classwise_dice(soft_out_seq_middle,soft_label_seq).cpu().numpy()
                gtv_dice_large = get_classwise_dice(soft_out_seq_large,soft_label_seq).cpu().numpy()
                gtv_dice_extra_middle = get_classwise_dice(soft_out_seq_extra_middle,soft_label_seq).cpu().numpy()
                gtv_dice_below_large = get_classwise_dice(soft_out_seq_below_large,soft_label_seq).cpu().numpy()

                gtv_dice_ensem_mid = get_classwise_dice(soft_out_ensem_mid,soft_label_seq).cpu().numpy()                
                gtv_dice_out_ave = get_classwise_dice(soft_out_ave_seq,soft_label_seq).cpu().numpy()
                gtv_dice_ensem_mid_uncert = get_classwise_dice(soft_out_ensem_mid_uncert,soft_label_seq).cpu().numpy()

                avg_test_dsc_small.append(gtv_dice_small)
                avg_test_dsc_middle.append(gtv_dice_middle)
                avg_test_dsc_large.append(gtv_dice_large)
                avg_test_dsc_extra_middle.append(gtv_dice_extra_middle)
                avg_test_dsc_below_large.append(gtv_dice_below_large)

                avg_test_dsc_ensem_mid.append(gtv_dice_ensem_mid)
                avg_test_dsc_out_ave.append(gtv_dice_out_ave)
                avg_test_dsc_ensem_mid_uncert.append(gtv_dice_ensem_mid_uncert)

                # if len(soft_out_seq) == 0:
                #     print('\n\n\n\n')
                #     print("$"*10)
                #     print("No prediction with --> {}".format(names))
                #     print("Present Classes --> {}".format(present_classes))
                #     print("$"*10)
                #     print('\n\n\n\n')
                #     continue

                print("Result in iter--{} for:{}, with Pixel-spacing:{}".format(iter_test,names,pixel_spacing))
                print("Result in iter--{} for:{}, with Pixel-spacing:{}".format(iter_test,names,pixel_spacing),file=infer_txt)
                print('Soft-out-shape',soft_out_seq_small.shape,'Soft Label-seq',soft_label_seq.shape)
                print('\n\n***************')
                for c in range(1,class_num):
                    print('Small-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_small[c]))
                    print('Small-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_small[c]),file=infer_txt)
                    print('Middle-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_middle[c]))
                    print('Middle-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_middle[c]),file=infer_txt)
                    print('Large-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_large[c]))
                    print('Large-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_large[c]),file=infer_txt)
                    print('Below-Large-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_below_large[c]))
                    print('Below-Large-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_below_large[c]),file=infer_txt)
                    print('Extra-Middle-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_extra_middle[c]))
                    print('Extra-Middle-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_extra_middle[c]),file=infer_txt)
                    print('\n****************** Ensemble Results ******************')
                    print('Ensemble Majority-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_ensem_mid[c]))
                    print('Ensemble Majority-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_ensem_mid[c]),file=infer_txt)
                    print('Aggregated Output-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_out_ave[c]))
                    print('Aggregated Output-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_out_ave[c]),file=infer_txt)
                    print('Ensemble Uncert-Majority-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_ensem_mid_uncert[c]))
                    print('Ensemble Uncert-Majority-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_ensem_mid_uncert[c]),file=infer_txt)

                print('Done Testing for iter-->{}'.format(iter_test))
                print('Done Testing for iter-->{}'.format(iter_test),file=infer_txt)
                print('\n\n\n')
                print('\n\n\n',file=infer_txt)


        print("\n\n\n")
        print("\n\n\n",file=infer_txt)
        print("*****"*10)
        print("*****"*10,file=infer_txt)

        avg_time = (time.time() - start_time) / len(self.test_loder)
        avg_test_dsc_small = np.asarray(avg_test_dsc_small).mean(axis = 0)
        avg_test_dsc_middle = np.asarray(avg_test_dsc_middle).mean(axis = 0)
        avg_test_dsc_large = np.asarray(avg_test_dsc_large).mean(axis = 0)
        avg_test_dsc_extra_middle = np.asarray(avg_test_dsc_extra_middle).mean(axis = 0)
        avg_test_dsc_below_large = np.asarray(avg_test_dsc_below_large).mean(axis = 0)

        avg_test_dsc_ensem_mid = np.asarray(avg_test_dsc_ensem_mid).mean(axis = 0)
        avg_test_dsc_out_ave = np.asarray(avg_test_dsc_out_ave).mean(axis=0)
        avg_test_dsc_ensem_mid_uncert = np.asarray(avg_test_dsc_ensem_mid_uncert).mean(axis = 0)

        print("average testing time {0:}".format(avg_time))
        print("Average DSC result for total iter--{}".format(iter_test+1))
        for c in range(1,class_num):
            print('Small--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_small[c]))
            print('Small--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_small[c]),file=infer_txt)            
            print('Middle--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_middle[c]))
            print('Middle--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_middle[c]),file=infer_txt)            
            print('Large--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_large[c]))
            print('Large--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_large[c]),file=infer_txt)            
            print('Extra-Middle--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_extra_middle[c]))
            print('Extra-Middle--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_extra_middle[c]),file=infer_txt)            
            print('Below-Large--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_below_large[c]))
            print('Below-Large--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_below_large[c]),file=infer_txt)            

            print('\n******************Average Ensemble Results ******************')
            print('Ave-Ensemble Majority-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_ensem_mid[c]))
            print('Ave-Ensemble Majority-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_ensem_mid[c]),file=infer_txt)
            print('Ave-Aggregated Output-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_out_ave[c]))
            print('Ave-Aggregated Output-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_out_ave[c]),file=infer_txt)
            print('Ave-Ensemble Uncert-Majority-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_ensem_mid_uncert[c]))
            print('Ave-Ensemble Uncert-Majority-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_ensem_mid_uncert[c]),file=infer_txt)

        print('*****************************************')
        print("Done Overall Testing")
        print('*****************************************\n\n\n')
        print('*****************************************',file=infer_txt)
        print("Done Overall Testing",file=infer_txt)
        print('*****************************************\n\n\n',file=infer_txt)


    def __surface_dist_eval(self,exp):

        if self.exp != 'debug':
            infer_txt = open( os.path.join(self.output_dir,'infer_output_{}.txt'.format(exp)) ,'a')
        else:
            infer_txt = open('infer_output_{}.txt'.format(exp) ,'w')
        device = torch.device(self.config['testing']['device_name'])

        checkpoint_small  = self.config['ensemble']['small']
        checkpoint_middle = self.config['ensemble']['middle']
        checkpoint_large = self.config['ensemble']['large']
        checkpoint_extra_middle = self.config['ensemble']['extra_middle']
        checkpoint_below_large = self.config['ensemble']['below_large']

        ckpt_small = torch.load(checkpoint_small)
        ckpt_middle = torch.load(checkpoint_middle)
        ckpt_large = torch.load(checkpoint_large)
        ckpt_extra_middle = torch.load(checkpoint_extra_middle)
        ckpt_below_large = torch.load(checkpoint_below_large)

        # self.checkpoint = torch.load(checkpoint)
        # self.net.load_state_dict(self.checkpoint['model_state_dict'])
        model_small = copy.deepcopy(self.net)
        model_middle = copy.deepcopy(self.net)
        model_large = copy.deepcopy(self.net)
        model_extra_middle = copy.deepcopy(self.net)
        model_below_large = copy.deepcopy(self.net)

        model_small.load_state_dict(ckpt_small['model_state_dict'])
        model_middle.load_state_dict(ckpt_middle['model_state_dict'])
        model_large.load_state_dict(ckpt_large['model_state_dict'])
        model_extra_middle.load_state_dict(ckpt_extra_middle['model_state_dict'])
        model_below_large.load_state_dict(ckpt_below_large['model_state_dict'])
        
        model_small.to(device)
        model_middle.to(device)
        model_large.to(device)
        model_extra_middle.to(device)
        model_below_large.to(device)

        model_small.eval()
        model_middle.eval()
        model_large.eval()
        model_extra_middle.eval()
        model_below_large.eval()

        print('Starting to test using checkpoint-Small:{}'.format(checkpoint_small))
        print('Starting to test using checkpoint-Small:{}'.format(checkpoint_small),file=infer_txt)

        print('Starting to test using checkpoint-Middle:{}'.format(checkpoint_middle))
        print('Starting to test using checkpoint-Middle:{}'.format(checkpoint_middle),file=infer_txt)

        print('Starting to test using checkpoint-Large:{}'.format(checkpoint_large))
        print('Starting to test using checkpoint-Large:{}'.format(checkpoint_large),file=infer_txt)

        print('Starting to test using checkpoint-Extra-Middle:{}'.format(checkpoint_extra_middle))
        print('Starting to test using checkpoint-Extra-Middle:{}'.format(checkpoint_extra_middle),file=infer_txt)

        print('Starting to test using checkpoint-Below-Large:{}'.format(checkpoint_below_large))
        print('Starting to test using checkpoint-Below-Large:{}'.format(checkpoint_below_large),file=infer_txt)

        class_num  = self.config['network']['class_num']
        SLICE = 16

        start_time = time.time()

        avg_test_iou_small = []
        avg_test_iou_middle = []
        avg_test_iou_large = []
        avg_test_iou_extra_middle = []
        avg_test_iou_below_large = []
        avg_test_iou_ensem_mid = []
        avg_test_iou_out_ave = []

        avg_test_assd_small = []
        avg_test_assd_middle = []
        avg_test_assd_large = []
        avg_test_assd_extra_middle = []
        avg_test_assd_below_large = []
        avg_test_assd_ensem_mid = []
        avg_test_assd_out_ave = []

        avg_test_hausdorff_small = []
        avg_test_hausdorff_middle = []
        avg_test_hausdorff_large = []
        avg_test_hausdorff_extra_middle = []
        avg_test_hausdorff_below_large = []
        avg_test_hausdorff_ensem_mid = []
        avg_test_hausdorff_out_ave = []


        avg_test_rve_small = []
        avg_test_rve_middle = []
        avg_test_rve_large = []
        avg_test_rve_extra_middle = []
        avg_test_rve_below_large = []
        avg_test_rve_ensem_mid = []
        avg_test_rve_out_ave = []


        gtv_variance_np_dir = 'results/gtv_maps2'
        os.makedirs(gtv_variance_np_dir,exist_ok=True)
        soft_max_f = nn.Softmax(dim=1)

        orig_data_dir = '../../Dataset_Rad2/nifti_gtv_test'
        with torch.no_grad():
            for iter_test,data in enumerate(self.test_loder):
                images,labels = data['image'],data['label']
                present_classes = torch.unique(labels.flatten()).numpy()
                gt_volume = labels.clone()
                images, labels = images.to(device), labels.to(device)
                total_gtv_voxels = torch.sum(labels.flatten()).item() 
                names  = data['names']
                nifti_name = os.path.join(orig_data_dir,names[0].replace('data','image'))
                img_nifti, pixel_spacing = load_origin_nifty_volume_as_array(nifti_name)
                img_nifti = torch.from_numpy(np.expand_dims(np.expand_dims(img_nifti,axis=0),axis=0)).to(device)
                print("Processing --> {} with input and mask shape --> {}, {} at Iteration--{}".format(names,images.shape,labels.shape,iter_test))
                print("Processing --> {} with input and mask shape --> {}, {} at Iteration--{}".format(names,images.shape,labels.shape,iter_test),file=infer_txt)
                needed_num_slice = SLICE - int(images.shape[2])%SLICE
                last_inp_slice,last_label_slice = images[0,0,-1,:,:],labels[0,0,-1,:,:]
                inp_slice = last_inp_slice.repeat(1,1,needed_num_slice,1,1)
                label_slice_extra = last_label_slice.repeat(1,1,needed_num_slice,1,1)
                images = torch.cat((images,inp_slice),dim=2)
                labels = torch.cat((labels,label_slice_extra),dim=2)
                img_nifti = torch.cat((img_nifti,inp_slice),dim=2)

                images = torch.split(images,SLICE,2)
                labels = torch.split(labels,SLICE,2)
                img_nifti = torch.split(img_nifti,SLICE,2)

                soft_out_seq_small = []
                soft_out_seq_middle = []
                soft_out_seq_large = []
                soft_out_seq_below_large = []
                soft_out_seq_extra_middle = []
                soft_out_ave_seq = []
                
                soft_label_seq = []
                input_slice_seq = []

                for idx,(inputs_slice,labels_slice) in enumerate(zip(images,labels)):
                    img_nifti_slice = img_nifti[idx]
                    # print(img_nifti_slice.shape)

                    if len(torch.unique(labels_slice))<class_num:
                        continue
                    total_slice_gtv_voxels = torch.sum(labels_slice.flatten()).cpu().item()
                    ratio_slice_gtv_voxels = round(total_slice_gtv_voxels/total_gtv_voxels,4)
                    
                    soft_y = get_soft_label(labels_slice,class_num,device)


                    #Small
                    output_small = model_small(inputs_slice)
                    outputs_argmax_small = torch.argmax(output_small, dim = 1, keepdim = True)
                    
                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_small = outputs_argmax_small[:,:,:end_slice,:,:]
                        soft_y = soft_y[:,:,:end_slice,:,:]
                        img_nifti_slice = img_nifti_slice[:,:,:end_slice,:,:]


                    soft_out_small  = get_soft_label(outputs_argmax_small, class_num,device)
                    soft_out_seq_small.append(soft_out_small)
                    soft_label_seq.append(soft_y)                        
                    input_slice_seq.append(img_nifti_slice)

                    
                    #Middle
                    output_middle = model_middle(inputs_slice)
                    outputs_argmax_middle = torch.argmax(output_middle, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_middle = outputs_argmax_middle[:,:,:end_slice,:,:]

                    soft_out_middle  = get_soft_label(outputs_argmax_middle, class_num,device)
                    soft_out_seq_middle.append(soft_out_middle)

                    #Large
                    output_large = model_large(inputs_slice)
                    outputs_argmax_large = torch.argmax(output_large, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_large = outputs_argmax_large[:,:,:end_slice,:,:]

                    soft_out_large  = get_soft_label(outputs_argmax_large, class_num,device)
                    soft_out_seq_large.append(soft_out_large)

                    #Below Large
                    output_below_large = model_below_large(inputs_slice)
                    outputs_argmax_below_large = torch.argmax(output_below_large, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_below_large = outputs_argmax_below_large[:,:,:end_slice,:,:]

                    soft_out_below_large  = get_soft_label(outputs_argmax_below_large, class_num,device)
                    soft_out_seq_below_large.append(soft_out_below_large)

                    #Extra Middle
                    output_extra_middle = model_extra_middle(inputs_slice)
                    outputs_argmax_extra_middle = torch.argmax(output_extra_middle, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_extra_middle = outputs_argmax_extra_middle[:,:,:end_slice,:,:]

                    soft_out_extra_middle  = get_soft_label(outputs_argmax_extra_middle, class_num,device)
                    soft_out_seq_extra_middle.append(soft_out_extra_middle)
                    
                    out_small_smax = soft_max_f(output_small)
                    out_middle_smax = soft_max_f(output_middle)
                    out_large_smax = soft_max_f(output_large)
                    out_extra_middle_smax = soft_max_f(output_extra_middle)
                    out_below_large_smax = soft_max_f(output_below_large) 
                    
                    ave_output = (out_middle_smax+out_small_smax+out_large_smax+out_extra_middle_smax+out_below_large_smax)/5
                    ave_output_argmax = torch.argmax(ave_output,dim=1,keepdim=True)
                    soft_out_ave = get_soft_label(ave_output_argmax, class_num,device)
                    
                    soft_out_ave_seq.append(soft_out_ave)


                    del output_extra_middle
                    del outputs_argmax_extra_middle
                    del output_small
                    del outputs_argmax_small
                    del output_middle
                    del outputs_argmax_middle
                    del output_large
                    del outputs_argmax_large
                    del output_below_large
                    del outputs_argmax_below_large

                #Input 
                soft_label_seq = torch.cat(soft_label_seq,dim=2).cpu()
                input_slice_seq = torch.cat(input_slice_seq,dim=2).cpu()

                #Output
                soft_out_seq_small = torch.cat(soft_out_seq_small,dim=2).cpu()
                soft_out_seq_middle = torch.cat(soft_out_seq_middle,dim=2).cpu()
                soft_out_seq_large = torch.cat(soft_out_seq_large,dim=2).cpu()
                soft_out_seq_extra_middle = torch.cat(soft_out_seq_extra_middle,dim=2).cpu()
                soft_out_seq_below_large = torch.cat(soft_out_seq_below_large,dim=2).cpu()
                
                #Average
                soft_out_ave_seq = torch.cat(soft_out_ave_seq,dim=2).cpu()



                summed_soft_out = soft_out_seq_large+soft_out_seq_middle+soft_out_seq_small+soft_out_seq_below_large+soft_out_seq_extra_middle
                soft_out_ensem_zeros = torch.zeros_like(soft_label_seq)
                soft_out_ensem_ones = torch.ones_like(soft_label_seq)
                soft_out_ensem_mid = torch.where(summed_soft_out>=3,soft_out_ensem_ones,soft_out_ensem_zeros)
                
                vol_groudtruth = torch.argmax(soft_label_seq, dim = 1, keepdim = True).squeeze().numpy()

                vol_small = torch.argmax(soft_out_seq_small, dim = 1, keepdim = True).squeeze().numpy()
                vol_middle = torch.argmax(soft_out_seq_middle, dim = 1, keepdim = True).squeeze().numpy()
                vol_large = torch.argmax(soft_out_seq_large, dim = 1, keepdim = True).squeeze().numpy()
                vol_below_large = torch.argmax(soft_out_seq_below_large, dim = 1, keepdim = True).squeeze().numpy()
                vol_extra_middle = torch.argmax(soft_out_seq_extra_middle, dim = 1, keepdim = True).squeeze().numpy()
                

                vol_ensem_mid = torch.argmax(soft_out_ensem_mid, dim = 1, keepdim = True).squeeze().numpy()
                vol_aggr_avg = torch.argmax(soft_out_ave_seq, dim = 1, keepdim = True).squeeze().numpy()


                #IOU Calculation
                st = time.time()
                iou_small = binary_iou(vol_small,vol_groudtruth)
                iou_middle = binary_iou(vol_middle,vol_groudtruth)
                iou_large = binary_iou(vol_large,vol_groudtruth)
                iou_below_large = binary_iou(vol_below_large,vol_groudtruth)
                iou_extra_middle = binary_iou(vol_extra_middle,vol_groudtruth)
                iou_ensem_mid = binary_iou(vol_ensem_mid,vol_groudtruth)
                iou_agg_avg = binary_iou(vol_aggr_avg,vol_groudtruth)

                avg_test_iou_small.append(iou_small)
                avg_test_iou_middle.append(iou_middle)
                avg_test_iou_large.append(iou_large)
                avg_test_iou_extra_middle.append(iou_extra_middle)
                avg_test_iou_below_large.append(iou_below_large)
                avg_test_iou_ensem_mid.append(iou_ensem_mid)
                avg_test_iou_out_ave.append(iou_agg_avg)
                print("Time taken To Calculate IOU:{} seconds".format(time.time()-st))


                #ASSD Calculation
                st = time.time()
                assd_small = binary_assd(vol_small,vol_groudtruth,pixel_spacing)
                assd_middle = binary_assd(vol_middle,vol_groudtruth,pixel_spacing)
                assd_large = binary_assd(vol_large,vol_groudtruth,pixel_spacing)
                assd_below_large = binary_assd(vol_below_large,vol_groudtruth,pixel_spacing)
                assd_extra_middle = binary_assd(vol_extra_middle,vol_groudtruth,pixel_spacing)
                assd_ensem_mid = binary_assd(vol_ensem_mid,vol_groudtruth,pixel_spacing)
                assd_agg_avg = binary_assd(vol_aggr_avg,vol_groudtruth,pixel_spacing)

                avg_test_assd_small.append(assd_small)
                avg_test_assd_middle.append(assd_middle)
                avg_test_assd_large.append(assd_large)
                avg_test_assd_extra_middle.append(assd_extra_middle)
                avg_test_assd_below_large.append(assd_below_large)
                avg_test_assd_ensem_mid.append(assd_ensem_mid)
                avg_test_assd_out_ave.append(assd_agg_avg)
                print("Time taken To Calculate ASSD:{} seconds".format(time.time()-st))

                #hausdorff Calculation
                st = time.time()
                hausdorff_small = binary_hausdorff(vol_small,vol_groudtruth,pixel_spacing)
                hausdorff_middle = binary_hausdorff(vol_middle,vol_groudtruth,pixel_spacing)
                hausdorff_large = binary_hausdorff(vol_large,vol_groudtruth,pixel_spacing)
                hausdorff_below_large = binary_hausdorff(vol_below_large,vol_groudtruth,pixel_spacing)
                hausdorff_extra_middle = binary_hausdorff(vol_extra_middle,vol_groudtruth,pixel_spacing)
                hausdorff_ensem_mid = binary_hausdorff(vol_ensem_mid,vol_groudtruth,pixel_spacing)
                hausdorff_agg_avg = binary_hausdorff(vol_aggr_avg,vol_groudtruth,pixel_spacing)

                avg_test_hausdorff_small.append(hausdorff_small)
                avg_test_hausdorff_middle.append(hausdorff_middle)
                avg_test_hausdorff_large.append(hausdorff_large)
                avg_test_hausdorff_extra_middle.append(hausdorff_extra_middle)
                avg_test_hausdorff_below_large.append(hausdorff_below_large)
                avg_test_hausdorff_ensem_mid.append(hausdorff_ensem_mid)
                avg_test_hausdorff_out_ave.append(hausdorff_agg_avg)
                print("Time taken To Calculate Hausdorff:{} seconds".format(time.time()-st))

                #RVE Calculation
                st = time.time()
                rve_small = binary_relative_volume_error(vol_small,vol_groudtruth)
                rve_middle = binary_relative_volume_error(vol_middle,vol_groudtruth)
                rve_large = binary_relative_volume_error(vol_large,vol_groudtruth)
                rve_below_large = binary_relative_volume_error(vol_below_large,vol_groudtruth)
                rve_extra_middle = binary_relative_volume_error(vol_extra_middle,vol_groudtruth)
                rve_ensem_mid = binary_relative_volume_error(vol_ensem_mid,vol_groudtruth)
                rve_agg_avg = binary_relative_volume_error(vol_aggr_avg,vol_groudtruth)

                avg_test_rve_small.append(rve_small)
                avg_test_rve_middle.append(rve_middle)
                avg_test_rve_large.append(rve_large)
                avg_test_rve_extra_middle.append(rve_extra_middle)
                avg_test_rve_below_large.append(rve_below_large)
                avg_test_rve_ensem_mid.append(rve_ensem_mid)
                avg_test_rve_out_ave.append(rve_agg_avg)
                print("Time taken To Calculate RVE:{} seconds".format(time.time()-st))


                print("Result in iter--{} for:{}, with Pixel-spacing:{}".format(iter_test,names,pixel_spacing))
                print("Result in iter--{} for:{}, with Pixel-spacing:{}".format(iter_test,names,pixel_spacing),file=infer_txt)
                print('Soft-out-shape',soft_out_seq_small.shape,'Soft Label-seq',soft_label_seq.shape)

                print('\n\n****************** IOU Results ******************')                
                print('Small-class_{}_dice,Test_IOU_value:{}'.format(1,iou_small))
                print('Middle-class_{}_dice,Test_IOU_value:{}'.format(1,iou_middle))
                print('Large-class_{}_dice,Test_IOU_value:{}'.format(1,iou_large))
                print('Below-Large-class_{}_dice,Test_IOU_value:{}'.format(1,iou_below_large))
                print('Extra-Middle-class_{}_dice,Test_IOU_value:{}'.format(1,iou_extra_middle))
                print('Ensemble Majority-class_{}_dice,Test_IOU_value:{}'.format(1,iou_ensem_mid))
                print('Aggregated Output-class_{}_dice,Test_IOU_value:{}'.format(1,iou_agg_avg))
                print('****************** IOU Results ******************')                

                print('\n\n****************** ASSD Results ******************')                
                print('Small-class_{}_dice,Test_IOU_value:{}'.format(1,assd_small))
                print('Middle-class_{}_dice,Test_IOU_value:{}'.format(1,assd_middle))
                print('Large-class_{}_dice,Test_IOU_value:{}'.format(1,assd_large))
                print('Below-Large-class_{}_dice,Test_IOU_value:{}'.format(1,assd_below_large))
                print('Extra-Middle-class_{}_dice,Test_IOU_value:{}'.format(1,assd_extra_middle))
                print('Ensemble Majority-class_{}_dice,Test_IOU_value:{}'.format(1,assd_ensem_mid))
                print('Aggregated Output-class_{}_dice,Test_IOU_value:{}'.format(1,assd_agg_avg))
                print('****************** IOU Results ******************')                


                print('\n\n****************** Hausdorff Results ******************')                
                print('Small-class_{}_dice,Test_IOU_value:{}'.format(1,hausdorff_small))
                print('Middle-class_{}_dice,Test_IOU_value:{}'.format(1,hausdorff_middle))
                print('Large-class_{}_dice,Test_IOU_value:{}'.format(1,hausdorff_large))
                print('Below-Large-class_{}_dice,Test_IOU_value:{}'.format(1,hausdorff_below_large))
                print('Extra-Middle-class_{}_dice,Test_IOU_value:{}'.format(1,hausdorff_extra_middle))
                print('Ensemble Majority-class_{}_dice,Test_IOU_value:{}'.format(1,hausdorff_ensem_mid))
                print('Aggregated Output-class_{}_dice,Test_IOU_value:{}'.format(1,hausdorff_agg_avg))
                print('****************** IOU Results ******************')                

                print('\n\n****************** RVE Results ******************')                
                print('Small-class_{}_dice,Test_IOU_value:{}'.format(1,rve_small))
                print('Middle-class_{}_dice,Test_IOU_value:{}'.format(1,rve_middle))
                print('Large-class_{}_dice,Test_IOU_value:{}'.format(1,rve_large))
                print('Below-Large-class_{}_dice,Test_IOU_value:{}'.format(1,rve_below_large))
                print('Extra-Middle-class_{}_dice,Test_IOU_value:{}'.format(1,rve_extra_middle))
                print('Ensemble Majority-class_{}_dice,Test_IOU_value:{}'.format(1,rve_ensem_mid))
                print('Aggregated Output-class_{}_dice,Test_IOU_value:{}'.format(1,rve_agg_avg))
                print('****************** IOU Results ******************')                

                print('Done Testing for iter-->{}'.format(iter_test))
                print('Done Testing for iter-->{}'.format(iter_test),file=infer_txt)
                print('\n\n\n')
                print('\n\n\n',file=infer_txt)

        #Average IOU Test Results
        avg_test_iou_small = sum(avg_test_iou_small)/len(avg_test_iou_small)
        avg_test_iou_middle = sum(avg_test_iou_middle)/len(avg_test_iou_middle)
        avg_test_iou_large = sum(avg_test_iou_large)/len(avg_test_iou_large)
        avg_test_iou_extra_middle = sum(avg_test_iou_extra_middle)/len(avg_test_iou_extra_middle)
        avg_test_iou_below_large = sum(avg_test_iou_below_large)/len(avg_test_iou_below_large)
        avg_test_iou_ensem_mid = sum(avg_test_iou_ensem_mid)/len(avg_test_iou_ensem_mid)
        avg_test_iou_out_ave = sum(avg_test_iou_out_ave)/len(avg_test_iou_out_ave)

        #Average ASSD Test Results
        avg_test_assd_small = sum(avg_test_assd_small)/len(avg_test_assd_small)
        avg_test_assd_middle = sum(avg_test_assd_middle)/len(avg_test_assd_middle)
        avg_test_assd_large = sum(avg_test_assd_large)/len(avg_test_assd_large)
        avg_test_assd_extra_middle = sum(avg_test_assd_extra_middle)/len(avg_test_assd_extra_middle)
        avg_test_assd_below_large = sum(avg_test_assd_below_large)/len(avg_test_assd_below_large)
        avg_test_assd_ensem_mid = sum(avg_test_assd_ensem_mid)/len(avg_test_assd_ensem_mid)
        avg_test_assd_out_ave = sum(avg_test_assd_out_ave)/len(avg_test_assd_out_ave)

        #Average Hausdorff Test Results
        avg_test_hausdorff_small = sum(avg_test_hausdorff_small)/len(avg_test_hausdorff_small)
        avg_test_hausdorff_middle = sum(avg_test_hausdorff_middle)/len(avg_test_hausdorff_middle)
        avg_test_hausdorff_large = sum(avg_test_hausdorff_large)/len(avg_test_hausdorff_large)
        avg_test_hausdorff_extra_middle = sum(avg_test_hausdorff_extra_middle)/len(avg_test_hausdorff_extra_middle)
        avg_test_hausdorff_below_large = sum(avg_test_hausdorff_below_large)/len(avg_test_hausdorff_below_large)
        avg_test_hausdorff_ensem_mid = sum(avg_test_hausdorff_ensem_mid)/len(avg_test_hausdorff_ensem_mid)
        avg_test_hausdorff_out_ave = sum(avg_test_hausdorff_out_ave)/len(avg_test_hausdorff_out_ave)

        #Average IOU Test Results
        avg_test_rve_small = sum(avg_test_rve_small)/len(avg_test_rve_small)
        avg_test_rve_middle = sum(avg_test_rve_middle)/len(avg_test_rve_middle)
        avg_test_rve_large = sum(avg_test_rve_large)/len(avg_test_rve_large)
        avg_test_rve_extra_middle = sum(avg_test_rve_extra_middle)/len(avg_test_rve_extra_middle)
        avg_test_rve_below_large = sum(avg_test_rve_below_large)/len(avg_test_rve_below_large)
        avg_test_rve_ensem_mid = sum(avg_test_rve_ensem_mid)/len(avg_test_rve_ensem_mid)
        avg_test_rve_out_ave = sum(avg_test_rve_out_ave)/len(avg_test_rve_out_ave)


        print("\n\n\n")
        print("\n\n\n",file=infer_txt)
        print("*****"*10)
        print("*****"*10,file=infer_txt)
        c = 1
        avg_time = (time.time() - start_time) / len(self.test_loder)

        print("average testing time {0:}".format(avg_time))
        print("Average Surface Distance Results for total iter--{}".format(iter_test+1))

        print('\n\n****************** Average Test IOU Results ******************')                
        print('Avg--Small-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_iou_small))
        print('Avg--Middle-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_iou_middle))
        print('Avg--Large-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_iou_large))
        print('Avg--Below-Large-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_iou_below_large))
        print('Avg--Extra-Middle-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_iou_extra_middle))
        print('Avg--Ensemble Majority-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_iou_ensem_mid))
        print('Avg--Aggregated Output-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_iou_out_ave))
        print('****************** Average Test IOU Results ******************\n\n')                

        print('\n\n****************** Average Test ASSD Results ******************')                
        print('Avg--Small-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_assd_small))
        print('Avg--Middle-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_assd_middle))
        print('Avg--Large-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_assd_large))
        print('Avg--Below-Large-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_assd_below_large))
        print('Avg--Extra-Middle-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_assd_extra_middle))
        print('Avg--Ensemble Majority-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_assd_ensem_mid))
        print('Avg--Aggregated Output-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_assd_out_ave))
        print('****************** Average Test ASSD Results ******************\n\n')                


        print('\n\n****************** Average Test Hausdorff Results ******************')                
        print('Avg--Small-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_hausdorff_small))
        print('Avg--Middle-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_hausdorff_middle))
        print('Avg--Large-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_hausdorff_large))
        print('Avg--Below-Large-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_hausdorff_below_large))
        print('Avg--Extra-Middle-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_hausdorff_extra_middle))
        print('Avg--Ensemble Majority-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_hausdorff_ensem_mid))
        print('Avg--Aggregated Output-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_hausdorff_out_ave))
        print('****************** Average Test Hausdorff Results ******************\n\n')                

        print('\n\n****************** Average Test RVE Results ******************')                
        print('Avg--Small-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_rve_small))
        print('Avg--Middle-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_rve_middle))
        print('Avg--Large-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_rve_large))
        print('Avg--Below-Large-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_rve_below_large))
        print('Avg--Extra-Middle-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_rve_extra_middle))
        print('Avg--Ensemble Majority-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_rve_ensem_mid))
        print('Avg--Aggregated Output-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_rve_out_ave))
        print('****************** Average Test RVE Results ******************\n\n')                


        print('*****************************************')
        print("Done Overall Testing")
        print('*****************************************\n\n\n')
        print('*****************************************',file=infer_txt)
        print("Done Overall Testing",file=infer_txt)
        print('*****************************************\n\n\n',file=infer_txt)


    def __sensi_speci_ppv(self,exp):

        if self.exp != 'debug':
            infer_txt = open( os.path.join(self.output_dir,'infer_output_{}.txt'.format(exp)) ,'a')
        else:
            infer_txt = open('infer_output_{}.txt'.format(exp) ,'w')
        device = torch.device(self.config['testing']['device_name'])

        checkpoint_small  = self.config['ensemble']['small']
        checkpoint_middle = self.config['ensemble']['middle']
        checkpoint_large = self.config['ensemble']['large']
        checkpoint_extra_middle = self.config['ensemble']['extra_middle']
        checkpoint_below_large = self.config['ensemble']['below_large']

        ckpt_small = torch.load(checkpoint_small)
        ckpt_middle = torch.load(checkpoint_middle)
        ckpt_large = torch.load(checkpoint_large)
        ckpt_extra_middle = torch.load(checkpoint_extra_middle)
        ckpt_below_large = torch.load(checkpoint_below_large)

        # self.checkpoint = torch.load(checkpoint)
        # self.net.load_state_dict(self.checkpoint['model_state_dict'])
        model_small = copy.deepcopy(self.net)
        model_middle = copy.deepcopy(self.net)
        model_large = copy.deepcopy(self.net)
        model_extra_middle = copy.deepcopy(self.net)
        model_below_large = copy.deepcopy(self.net)

        model_small.load_state_dict(ckpt_small['model_state_dict'])
        model_middle.load_state_dict(ckpt_middle['model_state_dict'])
        model_large.load_state_dict(ckpt_large['model_state_dict'])
        model_extra_middle.load_state_dict(ckpt_extra_middle['model_state_dict'])
        model_below_large.load_state_dict(ckpt_below_large['model_state_dict'])
        
        model_small.to(device)
        model_middle.to(device)
        model_large.to(device)
        model_extra_middle.to(device)
        model_below_large.to(device)

        model_small.eval()
        model_middle.eval()
        model_large.eval()
        model_extra_middle.eval()
        model_below_large.eval()
        
        patients_ppv_list = []
        patients_sensi_list = []
        patients_spec_list = []        

        print('Starting to test using checkpoint-Small:{}'.format(checkpoint_small))
        print('Starting to test using checkpoint-Small:{}'.format(checkpoint_small),file=infer_txt)

        print('Starting to test using checkpoint-Middle:{}'.format(checkpoint_middle))
        print('Starting to test using checkpoint-Middle:{}'.format(checkpoint_middle),file=infer_txt)

        print('Starting to test using checkpoint-Large:{}'.format(checkpoint_large))
        print('Starting to test using checkpoint-Large:{}'.format(checkpoint_large),file=infer_txt)

        print('Starting to test using checkpoint-Extra-Middle:{}'.format(checkpoint_extra_middle))
        print('Starting to test using checkpoint-Extra-Middle:{}'.format(checkpoint_extra_middle),file=infer_txt)

        print('Starting to test using checkpoint-Below-Large:{}'.format(checkpoint_below_large))
        print('Starting to test using checkpoint-Below-Large:{}'.format(checkpoint_below_large),file=infer_txt)

        class_num  = self.config['network']['class_num']
        SLICE = 16

        start_time = time.time()

        avg_test_sensi_small = []
        avg_test_sensi_middle = []
        avg_test_sensi_large = []
        avg_test_sensi_extra_middle = []
        avg_test_sensi_below_large = []
        avg_test_sensi_ensem_mid = []
        avg_test_sensi_out_ave = []

        avg_test_spec_small = []
        avg_test_spec_middle = []
        avg_test_spec_large = []
        avg_test_spec_extra_middle = []
        avg_test_spec_below_large = []
        avg_test_spec_ensem_mid = []
        avg_test_spec_out_ave = []

        avg_test_ppv_small = []
        avg_test_ppv_middle = []
        avg_test_ppv_large = []
        avg_test_ppv_extra_middle = []
        avg_test_ppv_below_large = []
        avg_test_ppv_ensem_mid = []
        avg_test_ppv_out_ave = []


        gtv_variance_np_dir = 'results/gtv_maps2'
        os.makedirs(gtv_variance_np_dir,exist_ok=True)
        soft_max_f = nn.Softmax(dim=1)

        orig_data_dir = '../../Dataset_Rad2/nifti_gtv_test'
        with torch.no_grad():
            for iter_test,data in enumerate(self.test_loder):
                images,labels = data['image'],data['label']
                present_classes = torch.unique(labels.flatten()).numpy()
                gt_volume = labels.clone()
                images, labels = images.to(device), labels.to(device)
                total_gtv_voxels = torch.sum(labels.flatten()).item() 
                names  = data['names']



                nifti_name = os.path.join(orig_data_dir,names[0].replace('data','image'))
                img_nifti, pixel_spacing = load_origin_nifty_volume_as_array(nifti_name)
                img_nifti = torch.from_numpy(np.expand_dims(np.expand_dims(img_nifti,axis=0),axis=0)).to(device)
                print("Processing --> {} with input and mask shape --> {}, {} at Iteration--{}".format(names,images.shape,labels.shape,iter_test))
                print("Processing --> {} with input and mask shape --> {}, {} at Iteration--{}".format(names,images.shape,labels.shape,iter_test),file=infer_txt)
                needed_num_slice = SLICE - int(images.shape[2])%SLICE
                last_inp_slice,last_label_slice = images[0,0,-1,:,:],labels[0,0,-1,:,:]
                inp_slice = last_inp_slice.repeat(1,1,needed_num_slice,1,1)
                label_slice_extra = last_label_slice.repeat(1,1,needed_num_slice,1,1)
                images = torch.cat((images,inp_slice),dim=2)
                labels = torch.cat((labels,label_slice_extra),dim=2)
                img_nifti = torch.cat((img_nifti,inp_slice),dim=2)

                images = torch.split(images,SLICE,2)
                labels = torch.split(labels,SLICE,2)
                img_nifti = torch.split(img_nifti,SLICE,2)

                soft_out_seq_small = []
                soft_out_seq_middle = []
                soft_out_seq_large = []
                soft_out_seq_below_large = []
                soft_out_seq_extra_middle = []
                soft_out_ave_seq = []
                
                soft_label_seq = []
                input_slice_seq = []

                for idx,(inputs_slice,labels_slice) in enumerate(zip(images,labels)):
                    img_nifti_slice = img_nifti[idx]
                    # print(img_nifti_slice.shape)

                    if len(torch.unique(labels_slice))<class_num:
                        continue
                    total_slice_gtv_voxels = torch.sum(labels_slice.flatten()).cpu().item()
                    ratio_slice_gtv_voxels = round(total_slice_gtv_voxels/total_gtv_voxels,4)
                    
                    soft_y = get_soft_label(labels_slice,class_num,device)


                    #Small
                    output_small = model_small(inputs_slice)
                    outputs_argmax_small = torch.argmax(output_small, dim = 1, keepdim = True)
                    
                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_small = outputs_argmax_small[:,:,:end_slice,:,:]
                        soft_y = soft_y[:,:,:end_slice,:,:]
                        img_nifti_slice = img_nifti_slice[:,:,:end_slice,:,:]


                    soft_out_small  = get_soft_label(outputs_argmax_small, class_num,device)
                    soft_out_seq_small.append(soft_out_small)
                    soft_label_seq.append(soft_y)                        
                    input_slice_seq.append(img_nifti_slice)

                    
                    #Middle
                    output_middle = model_middle(inputs_slice)
                    outputs_argmax_middle = torch.argmax(output_middle, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_middle = outputs_argmax_middle[:,:,:end_slice,:,:]

                    soft_out_middle  = get_soft_label(outputs_argmax_middle, class_num,device)
                    soft_out_seq_middle.append(soft_out_middle)

                    #Large
                    output_large = model_large(inputs_slice)
                    outputs_argmax_large = torch.argmax(output_large, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_large = outputs_argmax_large[:,:,:end_slice,:,:]

                    soft_out_large  = get_soft_label(outputs_argmax_large, class_num,device)
                    soft_out_seq_large.append(soft_out_large)

                    #Below Large
                    output_below_large = model_below_large(inputs_slice)
                    outputs_argmax_below_large = torch.argmax(output_below_large, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_below_large = outputs_argmax_below_large[:,:,:end_slice,:,:]

                    soft_out_below_large  = get_soft_label(outputs_argmax_below_large, class_num,device)
                    soft_out_seq_below_large.append(soft_out_below_large)

                    #Extra Middle
                    output_extra_middle = model_extra_middle(inputs_slice)
                    outputs_argmax_extra_middle = torch.argmax(output_extra_middle, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_extra_middle = outputs_argmax_extra_middle[:,:,:end_slice,:,:]

                    soft_out_extra_middle  = get_soft_label(outputs_argmax_extra_middle, class_num,device)
                    soft_out_seq_extra_middle.append(soft_out_extra_middle)
                    
                    out_small_smax = soft_max_f(output_small)
                    out_middle_smax = soft_max_f(output_middle)
                    out_large_smax = soft_max_f(output_large)
                    out_extra_middle_smax = soft_max_f(output_extra_middle)
                    out_below_large_smax = soft_max_f(output_below_large) 
                    
                    ave_output = (out_middle_smax+out_small_smax+out_large_smax+out_extra_middle_smax+out_below_large_smax)/5
                    ave_output_argmax = torch.argmax(ave_output,dim=1,keepdim=True)
                    soft_out_ave = get_soft_label(ave_output_argmax, class_num,device)
                    
                    soft_out_ave_seq.append(soft_out_ave)


                    del output_extra_middle
                    del outputs_argmax_extra_middle
                    del output_small
                    del outputs_argmax_small
                    del output_middle
                    del outputs_argmax_middle
                    del output_large
                    del outputs_argmax_large
                    del output_below_large
                    del outputs_argmax_below_large

                #Input 
                soft_label_seq = torch.cat(soft_label_seq,dim=2).cpu()
                input_slice_seq = torch.cat(input_slice_seq,dim=2).cpu()

                #Output
                soft_out_seq_small = torch.cat(soft_out_seq_small,dim=2).cpu()
                soft_out_seq_middle = torch.cat(soft_out_seq_middle,dim=2).cpu()
                soft_out_seq_large = torch.cat(soft_out_seq_large,dim=2).cpu()
                soft_out_seq_extra_middle = torch.cat(soft_out_seq_extra_middle,dim=2).cpu()
                soft_out_seq_below_large = torch.cat(soft_out_seq_below_large,dim=2).cpu()
                
                #Average
                soft_out_ave_seq = torch.cat(soft_out_ave_seq,dim=2).cpu()



                summed_soft_out = soft_out_seq_large+soft_out_seq_middle+soft_out_seq_small+soft_out_seq_below_large+soft_out_seq_extra_middle
                soft_out_ensem_zeros = torch.zeros_like(soft_label_seq)
                soft_out_ensem_ones = torch.ones_like(soft_label_seq)
                soft_out_ensem_mid = torch.where(summed_soft_out>=3,soft_out_ensem_ones,soft_out_ensem_zeros)
                
                vol_groudtruth = torch.argmax(soft_label_seq, dim = 1, keepdim = True).squeeze().numpy()

                vol_small = torch.argmax(soft_out_seq_small, dim = 1, keepdim = True).squeeze().numpy()
                vol_middle = torch.argmax(soft_out_seq_middle, dim = 1, keepdim = True).squeeze().numpy()
                vol_large = torch.argmax(soft_out_seq_large, dim = 1, keepdim = True).squeeze().numpy()
                vol_below_large = torch.argmax(soft_out_seq_below_large, dim = 1, keepdim = True).squeeze().numpy()
                vol_extra_middle = torch.argmax(soft_out_seq_extra_middle, dim = 1, keepdim = True).squeeze().numpy()
                

                vol_ensem_mid = torch.argmax(soft_out_ensem_mid, dim = 1, keepdim = True).squeeze().numpy()
                vol_aggr_avg = torch.argmax(soft_out_ave_seq, dim = 1, keepdim = True).squeeze().numpy()


                #Sensitivity Calculation
                st = time.time()
                sensi_small = sensitivity(vol_small,vol_groudtruth)
                sensi_middle = sensitivity(vol_middle,vol_groudtruth)
                sensi_large = sensitivity(vol_large,vol_groudtruth)
                sensi_below_large = sensitivity(vol_below_large,vol_groudtruth)
                sensi_extra_middle = sensitivity(vol_extra_middle,vol_groudtruth)
                sensi_ensem_mid = sensitivity(vol_ensem_mid,vol_groudtruth)
                sensi_agg_avg = sensitivity(vol_aggr_avg,vol_groudtruth)
                
                patients_sensi_list.append([names,sensi_small,sensi_middle,sensi_extra_middle,sensi_below_large,sensi_large,sensi_ensem_mid,sensi_agg_avg])

                avg_test_sensi_small.append(sensi_small)
                avg_test_sensi_middle.append(sensi_middle)
                avg_test_sensi_large.append(sensi_large)
                avg_test_sensi_extra_middle.append(sensi_extra_middle)
                avg_test_sensi_below_large.append(sensi_below_large)
                avg_test_sensi_ensem_mid.append(sensi_ensem_mid)
                avg_test_sensi_out_ave.append(sensi_agg_avg)
                print("Time taken To Calculate SENSITIVITY:{} seconds".format(time.time()-st))


                #Specificity Calculation
                st = time.time()
                spec_small = specificity(vol_small,vol_groudtruth,pixel_spacing)
                spec_middle = specificity(vol_middle,vol_groudtruth,pixel_spacing)
                spec_large = specificity(vol_large,vol_groudtruth,pixel_spacing)
                spec_below_large = specificity(vol_below_large,vol_groudtruth,pixel_spacing)
                spec_extra_middle = specificity(vol_extra_middle,vol_groudtruth,pixel_spacing)
                spec_ensem_mid = specificity(vol_ensem_mid,vol_groudtruth,pixel_spacing)
                spec_agg_avg = specificity(vol_aggr_avg,vol_groudtruth,pixel_spacing)
                
                patients_spec_list.append([names,spec_small,spec_middle,spec_extra_middle,spec_below_large,spec_large,spec_ensem_mid,spec_agg_avg])
                


                avg_test_spec_small.append(spec_small)
                avg_test_spec_middle.append(spec_middle)
                avg_test_spec_large.append(spec_large)
                avg_test_spec_extra_middle.append(spec_extra_middle)
                avg_test_spec_below_large.append(spec_below_large)
                avg_test_spec_ensem_mid.append(spec_ensem_mid)
                avg_test_spec_out_ave.append(spec_agg_avg)
                print("Time taken To Calculate SPECIFICITY:{} seconds".format(time.time()-st))

                #PPV Calculation
                st = time.time()
                ppv_small = ppv_func(vol_small,vol_groudtruth,pixel_spacing)
                ppv_middle = ppv_func(vol_middle,vol_groudtruth,pixel_spacing)
                ppv_large = ppv_func(vol_large,vol_groudtruth,pixel_spacing)
                ppv_below_large = ppv_func(vol_below_large,vol_groudtruth,pixel_spacing)
                ppv_extra_middle = ppv_func(vol_extra_middle,vol_groudtruth,pixel_spacing)
                ppv_ensem_mid = ppv_func(vol_ensem_mid,vol_groudtruth,pixel_spacing)
                ppv_agg_avg = ppv_func(vol_aggr_avg,vol_groudtruth,pixel_spacing)

                patients_ppv_list.append([names,ppv_small,ppv_middle,ppv_extra_middle,ppv_below_large,ppv_large,ppv_ensem_mid,ppv_agg_avg])

                avg_test_ppv_small.append(ppv_small)
                avg_test_ppv_middle.append(ppv_middle)
                avg_test_ppv_large.append(ppv_large)
                avg_test_ppv_extra_middle.append(ppv_extra_middle)
                avg_test_ppv_below_large.append(ppv_below_large)
                avg_test_ppv_ensem_mid.append(ppv_ensem_mid)
                avg_test_ppv_out_ave.append(ppv_agg_avg)
                print("Time taken To Calculate PPV:{} seconds".format(time.time()-st))


                print("Result in iter--{} for:{}, with Pixel-spacing:{}".format(iter_test,names,pixel_spacing))
                print("Result in iter--{} for:{}, with Pixel-spacing:{}".format(iter_test,names,pixel_spacing),file=infer_txt)
                print('Soft-out-shape',soft_out_seq_small.shape,'Soft Label-seq',soft_label_seq.shape)

                print('\n\n****************** Sensitivity Results ******************')                
                print('Small-class_{}_dice:{}'.format(1,sensi_small))
                print('Middle-class_{}_dice,:{}'.format(1,sensi_middle))
                print('Large-class_{}_dice,Test_sensi_value:{}'.format(1,sensi_large))
                print('Below-Large-class_{}_dice,:{}'.format(1,sensi_below_large))
                print('Extra-Middle-class_{}_dice,:{}'.format(1,sensi_extra_middle))
                print('Ensemble Majority-class_{}_dice,:{}'.format(1,sensi_ensem_mid))
                print('Aggregated Output-class_{}_dice,:{}'.format(1,sensi_agg_avg))
                print('****************** Sensi Results ******************')                

                print('\n\n****************** Specificity Results ******************')                
                print('Small-class_{}_dice,:{}'.format(1,spec_small))
                print('Middle-class_{}_dice,:{}'.format(1,spec_middle))
                print('Large-class_{}_dice,:{}'.format(1,spec_large))
                print('Below-Large-class_{}_dice,:{}'.format(1,spec_below_large))
                print('Extra-Middle-class_{}_dice,:{}'.format(1,spec_extra_middle))
                print('Ensemble Majority-class_{}_dice,:{}'.format(1,spec_ensem_mid))
                print('Aggregated Output-class_{}_dice,:{}'.format(1,spec_agg_avg))
                print('****************** SPecificity Results ******************')                


                print('\n\n****************** PPV Results ******************')                
                print('Small-class_{}_dice,:{}'.format(1,ppv_small))
                print('Middle-class_{}_dice,:{}'.format(1,ppv_middle))
                print('Large-class_{}_dice,:{}'.format(1,ppv_large))
                print('Below-Large-class_{}_dice,:{}'.format(1,ppv_below_large))
                print('Extra-Middle-class_{}_dice,:{}'.format(1,ppv_extra_middle))
                print('Ensemble Majority-class_{}_dice,:{}'.format(1,ppv_ensem_mid))
                print('Aggregated Output-class_{}_dice,:{}'.format(1,ppv_agg_avg))
                print('****************** PPV Results ******************')                


                print('Done Testing for iter-->{}'.format(iter_test))
                print('Done Testing for iter-->{}'.format(iter_test),file=infer_txt)
                print('\n\n\n')
                print('\n\n\n',file=infer_txt)

        #Average Sensitivity Test Results
        avg_test_sensi_small = sum(avg_test_sensi_small)/len(avg_test_sensi_small)
        avg_test_sensi_middle = sum(avg_test_sensi_middle)/len(avg_test_sensi_middle)
        avg_test_sensi_large = sum(avg_test_sensi_large)/len(avg_test_sensi_large)
        avg_test_sensi_extra_middle = sum(avg_test_sensi_extra_middle)/len(avg_test_sensi_extra_middle)
        avg_test_sensi_below_large = sum(avg_test_sensi_below_large)/len(avg_test_sensi_below_large)
        avg_test_sensi_ensem_mid = sum(avg_test_sensi_ensem_mid)/len(avg_test_sensi_ensem_mid)
        avg_test_sensi_out_ave = sum(avg_test_sensi_out_ave)/len(avg_test_sensi_out_ave)

        #Average Specificity Test Results
        avg_test_spec_small = sum(avg_test_spec_small)/len(avg_test_spec_small)
        avg_test_spec_middle = sum(avg_test_spec_middle)/len(avg_test_spec_middle)
        avg_test_spec_large = sum(avg_test_spec_large)/len(avg_test_spec_large)
        avg_test_spec_extra_middle = sum(avg_test_spec_extra_middle)/len(avg_test_spec_extra_middle)
        avg_test_spec_below_large = sum(avg_test_spec_below_large)/len(avg_test_spec_below_large)
        avg_test_spec_ensem_mid = sum(avg_test_spec_ensem_mid)/len(avg_test_spec_ensem_mid)
        avg_test_spec_out_ave = sum(avg_test_spec_out_ave)/len(avg_test_spec_out_ave)

        #Average PPV Test Results
        avg_test_ppv_small = sum(avg_test_ppv_small)/len(avg_test_ppv_small)
        avg_test_ppv_middle = sum(avg_test_ppv_middle)/len(avg_test_ppv_middle)
        avg_test_ppv_large = sum(avg_test_ppv_large)/len(avg_test_ppv_large)
        avg_test_ppv_extra_middle = sum(avg_test_ppv_extra_middle)/len(avg_test_ppv_extra_middle)
        avg_test_ppv_below_large = sum(avg_test_ppv_below_large)/len(avg_test_ppv_below_large)
        avg_test_ppv_ensem_mid = sum(avg_test_ppv_ensem_mid)/len(avg_test_ppv_ensem_mid)
        avg_test_ppv_out_ave = sum(avg_test_ppv_out_ave)/len(avg_test_ppv_out_ave)



        print("\n\n\n")
        print("\n\n\n",file=infer_txt)
        print("*****"*10)
        print("*****"*10,file=infer_txt)
        c = 1
        avg_time = (time.time() - start_time) / len(self.test_loder)

        print("average testing time {0:}".format(avg_time))
        print("Average Surface Distance Results for total iter--{}".format(iter_test+1))

        print('\n\n****************** Average Test Sensi Results ******************')                
        print('Avg--Small-class_{}_sensi,Test_sensi_value:{}'.format(c,avg_test_sensi_small))
        print('Avg--Middle-class_{}_sensi,Test_sensi_value:{}'.format(c,avg_test_sensi_middle))
        print('Avg--Large-class_{}_sensi,Test_sensi_value:{}'.format(c,avg_test_sensi_large))
        print('Avg--Below-Large-class_{}_sensi,Test_sensi_value:{}'.format(c,avg_test_sensi_below_large))
        print('Avg--Extra-Middle-class_{}_sensi,Test_sensi_value:{}'.format(c,avg_test_sensi_extra_middle))
        print('Avg--Ensemble Majority-class_{}_sensi,Test_sensi_value:{}'.format(c,avg_test_sensi_ensem_mid))
        print('Avg--Aggregated Output-class_{}_sensi,Test_sensi_value:{}'.format(c,avg_test_sensi_out_ave))
        print('****************** Average Test Sensi Results ******************\n\n')                

        print('\n\n****************** Average Test spec Results ******************')                
        print('Avg--Small-class_{}_spec,Test_spec_value:{}'.format(c,avg_test_spec_small))
        print('Avg--Middle-class_{}_spec,Test_spec_value:{}'.format(c,avg_test_spec_middle))
        print('Avg--Large-class_{}_spec,Test_spec_value:{}'.format(c,avg_test_spec_large))
        print('Avg--Below-Large-class_{}_spec,Test_spec_value:{}'.format(c,avg_test_spec_below_large))
        print('Avg--Extra-Middle-class_{}_spec,Test_spec_value:{}'.format(c,avg_test_spec_extra_middle))
        print('Avg--Ensemble Majority-class_{}_spec,Test_spec_value:{}'.format(c,avg_test_spec_ensem_mid))
        print('Avg--Aggregated Output-class_{}_spec,Test_spec_value:{}'.format(c,avg_test_spec_out_ave))
        print('****************** Average Test spec Results ******************\n\n')                


        print('\n\n****************** Average Test ppv Results ******************')                
        print('Avg--Small-class_{}_ppv,Test_ppv_value:{}'.format(c,avg_test_ppv_small))
        print('Avg--Middle-class_{}_ppv,Test_ppv_value:{}'.format(c,avg_test_ppv_middle))
        print('Avg--Large-class_{}_ppv,Test_ppv_value:{}'.format(c,avg_test_ppv_large))
        print('Avg--Below-Large-class_{}_ppv,Test_ppv_value:{}'.format(c,avg_test_ppv_below_large))
        print('Avg--Extra-Middle-class_{}_ppv,Test_ppv_value:{}'.format(c,avg_test_ppv_extra_middle))
        print('Avg--Ensemble Majority-class_{}_ppv,Test_ppv_value:{}'.format(c,avg_test_ppv_ensem_mid))
        print('Avg--Aggregated Output-class_{}_ppv,Test_ppv_value:{}'.format(c,avg_test_ppv_out_ave))
        print('****************** Average Test ppv Results ******************\n\n')                

        print('*****************************************')
        print("Done Overall Testing")
        print('*****************************************\n\n\n')
        print('*****************************************',file=infer_txt)
        print("Done Overall Testing",file=infer_txt)
        print('*****************************************\n\n\n',file=infer_txt)
        
        with open('results/sensi_out.csv','w',newline="") as f:
            writer_sensi = csv.writer(f)
            writer_sensi.writerows(patients_sensi_list)

        with open('results/spec_out.csv','w',newline="") as g:
            writer_spec = csv.writer(g)
            writer_spec.writerows(patients_spec_list)

        with open('results/ppv_out.csv','w',newline="") as h:
            writer_ppv = csv.writer(h)
            writer_ppv.writerows(patients_ppv_list)

    def __uncertainty_estimation(self,exp):

        if self.exp != 'debug':
            infer_txt = open( os.path.join(self.output_dir,'infer_output_{}.txt'.format(exp)) ,'a')
        else:
            infer_txt = open('infer_output_{}.txt'.format(exp) ,'w')
        device = torch.device(self.config['testing']['device_name'])

        checkpoint_small  = self.config['ensemble']['small']
        checkpoint_middle = self.config['ensemble']['middle']
        checkpoint_large = self.config['ensemble']['large']
        checkpoint_extra_middle = self.config['ensemble']['extra_middle']
        checkpoint_below_large = self.config['ensemble']['below_large']

        ckpt_small = torch.load(checkpoint_small)
        ckpt_middle = torch.load(checkpoint_middle)
        ckpt_large = torch.load(checkpoint_large)
        ckpt_extra_middle = torch.load(checkpoint_extra_middle)
        ckpt_below_large = torch.load(checkpoint_below_large)

        # self.checkpoint = torch.load(checkpoint)
        # self.net.load_state_dict(self.checkpoint['model_state_dict'])
        model_small = copy.deepcopy(self.net)
        model_middle = copy.deepcopy(self.net)
        model_large = copy.deepcopy(self.net)
        model_extra_middle = copy.deepcopy(self.net)
        model_below_large = copy.deepcopy(self.net)

        model_small.load_state_dict(ckpt_small['model_state_dict'])
        model_middle.load_state_dict(ckpt_middle['model_state_dict'])
        model_large.load_state_dict(ckpt_large['model_state_dict'])
        model_extra_middle.load_state_dict(ckpt_extra_middle['model_state_dict'])
        model_below_large.load_state_dict(ckpt_below_large['model_state_dict'])
        
        model_small.to(device)
        model_middle.to(device)
        model_large.to(device)
        model_extra_middle.to(device)
        model_below_large.to(device)

        model_small.eval()
        model_middle.eval()
        model_large.eval()
        model_extra_middle.eval()
        model_below_large.eval()

        print('Starting to test using checkpoint-Small:{}'.format(checkpoint_small))
        print('Starting to test using checkpoint-Small:{}'.format(checkpoint_small),file=infer_txt)

        print('Starting to test using checkpoint-Middle:{}'.format(checkpoint_middle))
        print('Starting to test using checkpoint-Middle:{}'.format(checkpoint_middle),file=infer_txt)

        print('Starting to test using checkpoint-Large:{}'.format(checkpoint_large))
        print('Starting to test using checkpoint-Large:{}'.format(checkpoint_large),file=infer_txt)

        print('Starting to test using checkpoint-Extra-Middle:{}'.format(checkpoint_extra_middle))
        print('Starting to test using checkpoint-Extra-Middle:{}'.format(checkpoint_extra_middle),file=infer_txt)

        print('Starting to test using checkpoint-Below-Large:{}'.format(checkpoint_below_large))
        print('Starting to test using checkpoint-Below-Large:{}'.format(checkpoint_below_large),file=infer_txt)

        class_num  = self.config['network']['class_num']
        SLICE = 24

        start_time = time.time()
        avg_test_dsc_small = []
        avg_test_dsc_middle = []
        avg_test_dsc_large = []
        avg_test_dsc_extra_middle = []
        avg_test_dsc_below_large = []
        avg_test_dsc_ensem_mid = []
        avg_test_dsc_out_ave = []
        


        gtv_variance_np_dir = 'results/feb22_gtv_uncertainty_map2'
        os.makedirs(gtv_variance_np_dir,exist_ok=True)
        soft_max_f = nn.Softmax(dim=1)

        orig_data_dir = '../../Dataset_Rad2/nifti_gtv_test'
        with torch.no_grad():
            for iter_test,data in enumerate(self.test_loder):

                images,labels = data['image'],data['label']
                present_classes = torch.unique(labels.flatten()).numpy()
                images, labels = images.to(device), labels.to(device)
                total_gtv_voxels = torch.sum(labels.flatten()).item() 
                names  = data['names']
                nifti_name = os.path.join(orig_data_dir,names[0].replace('data','image'))
                img_nifti, pixel_spacing = load_origin_nifty_volume_as_array(nifti_name)
                img_nifti = torch.from_numpy(np.expand_dims(np.expand_dims(img_nifti,axis=0),axis=0)).to(device)
                # print(img_nifti.shape,images.shape,labels.shape)
                # pixel_spacing = data['spacing']
                print("Processing --> {} with input and mask shape --> {}, {} at Iteration--{}".format(names,images.shape,labels.shape,iter_test))
                print("Processing --> {} with input and mask shape --> {}, {} at Iteration--{}".format(names,images.shape,labels.shape,iter_test),file=infer_txt)
                needed_num_slice = SLICE - int(images.shape[2])%SLICE
                last_inp_slice,last_label_slice = images[0,0,-1,:,:],labels[0,0,-1,:,:]
                inp_slice = last_inp_slice.repeat(1,1,needed_num_slice,1,1)
                label_slice_extra = last_label_slice.repeat(1,1,needed_num_slice,1,1)
                images = torch.cat((images,inp_slice),dim=2)
                labels = torch.cat((labels,label_slice_extra),dim=2)
                img_nifti = torch.cat((img_nifti,inp_slice),dim=2)

                images = torch.split(images,SLICE,2)
                labels = torch.split(labels,SLICE,2)
                img_nifti = torch.split(img_nifti,SLICE,2)

                soft_out_seq_small = []
                soft_out_seq_middle = []
                soft_out_seq_large = []
                soft_out_seq_below_large = []
                soft_out_seq_extra_middle = []
                soft_out_ave_seq = []
                first_idx = None
                last_idx = None
                soft_label_seq = []
                input_slice_seq = []
                
                images_input_list = []
                labels_input_list = []
                end_slice = SLICE-needed_num_slice

                for idx,(inputs_slice,labels_slice) in enumerate(zip(images,labels)):
                    if len(torch.unique(labels_slice))<class_num:
                        continue
                    img_nifti_slice = img_nifti[idx]
                    labels_input = labels_slice
                    names_gtv_np = names[0].split('/')[-1].strip('.nii')
                    out_npz_path = os.path.join(gtv_variance_np_dir,names_gtv_np)
                    if os.path.isfile(out_npz_path):
                        continue

                    if idx == (len(images)-1):
                        img_nifti_slice = img_nifti_slice[:,:,:end_slice,:,:]
                        labels_input = labels_input[:,:,:end_slice,:,:]

                    labels_input_list.append(labels_input.cpu())
                    images_input_list.append(img_nifti_slice)
                    
                    if len(torch.unique(labels_slice))==class_num:
                        if first_idx == None:
                            first_idx = idx
                    else:
                        if first_idx != None and last_idx == None:
                            last_idx = idx
                    
                    soft_y = get_soft_label(labels_slice,class_num,device).cpu()
                    total_slice_gtv_voxels = torch.sum(labels_slice.flatten()).cpu().item()
                    ratio_slice_gtv_voxels = round(total_slice_gtv_voxels/total_gtv_voxels,4)

                    #Small
                    output_small = model_small(inputs_slice)
                    outputs_argmax_small = torch.argmax(output_small, dim = 1, keepdim = True)
                    
                    if idx == (len(images)-1):
                        outputs_argmax_small = outputs_argmax_small[:,:,:end_slice,:,:]
                        soft_y = soft_y[:,:,:end_slice,:,:]


                    soft_out_small  = get_soft_label(outputs_argmax_small, class_num,device).cpu()
                    soft_out_seq_small.append(soft_out_small)
                    soft_label_seq.append(soft_y)                        

                    
                    #Middle
                    output_middle = model_middle(inputs_slice)
                    outputs_argmax_middle = torch.argmax(output_middle, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        outputs_argmax_middle = outputs_argmax_middle[:,:,:end_slice,:,:]

                    soft_out_middle  = get_soft_label(outputs_argmax_middle, class_num,device).cpu()
                    soft_out_seq_middle.append(soft_out_middle)

                    #Large
                    output_large = model_large(inputs_slice)
                    outputs_argmax_large = torch.argmax(output_large, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        outputs_argmax_large = outputs_argmax_large[:,:,:end_slice,:,:]

                    soft_out_large  = get_soft_label(outputs_argmax_large, class_num,device).cpu()
                    soft_out_seq_large.append(soft_out_large)

                    #Below Large
                    output_below_large = model_below_large(inputs_slice)
                    outputs_argmax_below_large = torch.argmax(output_below_large, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        outputs_argmax_below_large = outputs_argmax_below_large[:,:,:end_slice,:,:]

                    soft_out_below_large  = get_soft_label(outputs_argmax_below_large, class_num,device).cpu()
                    soft_out_seq_below_large.append(soft_out_below_large)

                    #Extra Middle
                    output_extra_middle = model_extra_middle(inputs_slice)
                    outputs_argmax_extra_middle = torch.argmax(output_extra_middle, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        outputs_argmax_extra_middle = outputs_argmax_extra_middle[:,:,:end_slice,:,:]

                    soft_out_extra_middle  = get_soft_label(outputs_argmax_extra_middle, class_num,device).cpu()
                    soft_out_seq_extra_middle.append(soft_out_extra_middle)
                    

                    out_small_smax = soft_max_f(output_small.cpu())
                    out_middle_smax = soft_max_f(output_middle.cpu())
                    out_large_smax = soft_max_f(output_large.cpu())
                    out_extra_middle_smax = soft_max_f(output_extra_middle.cpu())
                    out_below_large_smax = soft_max_f(output_below_large.cpu()) 
                    
                    ave_output = (out_middle_smax+out_small_smax+out_large_smax+out_extra_middle_smax+out_below_large_smax)/5
                    ave_output_argmax = torch.argmax(ave_output,dim=1,keepdim=True)
                    if idx == (len(images)-1):
                        ave_output_argmax = ave_output_argmax[:,:,:end_slice,:,:]

                    soft_out_ave = get_soft_label(ave_output_argmax, class_num,device)
                    soft_out_ave_seq.append(soft_out_ave)


                    del output_extra_middle
                    del outputs_argmax_extra_middle
                    del output_small
                    del outputs_argmax_small
                    del output_middle
                    del outputs_argmax_middle
                    del output_large
                    del outputs_argmax_large
                    del output_below_large
                    del outputs_argmax_below_large


                    dsc_value_iter_small = get_classwise_dice(soft_out_small, soft_y).cpu().numpy()
                    dsc_value_iter_middle = get_classwise_dice(soft_out_middle, soft_y).cpu().numpy()
                    dsc_value_iter_large = get_classwise_dice(soft_out_large, soft_y).cpu().numpy()
                    dsc_value_iter_extra_middle = get_classwise_dice(soft_out_extra_middle, soft_y).cpu().numpy()
                    dsc_value_iter_below_large = get_classwise_dice(soft_out_below_large, soft_y).cpu().numpy()
                    dsc_value_iter_ave = get_classwise_dice(soft_out_ave, soft_y).cpu().numpy()



                    # print("Small--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_small,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels))
                    # print("Small--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_small,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels),file=infer_txt)

                    # print("Middle--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_middle,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels))
                    # print("Middle--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_middle,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels),file=infer_txt)

                    # print("Large--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_large,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels))
                    # print("Large--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_large,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels),file=infer_txt)

                    # print("Extra-Middle--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_extra_middle,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels))
                    # print("Extra-Middle--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_extra_middle,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels),file=infer_txt)

                    # print("Below-Large--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_below_large,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels))
                    # print("Below-Large--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_below_large,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels),file=infer_txt)

                    # print("Aggregated Output--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_ave,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels))
                    # print("Aggregated Output--DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter_ave,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels),file=infer_txt)

                #Input 
                soft_label_seq = torch.cat(soft_label_seq,dim=2)
                input_slice_seq = torch.cat(images_input_list,dim=2)

                #Output
                soft_out_seq_small = torch.cat(soft_out_seq_small,dim=2)
                soft_out_seq_middle = torch.cat(soft_out_seq_middle,dim=2)
                soft_out_seq_large = torch.cat(soft_out_seq_large,dim=2)
                soft_out_seq_extra_middle = torch.cat(soft_out_seq_extra_middle,dim=2)
                soft_out_seq_below_large = torch.cat(soft_out_seq_below_large,dim=2)                
                #Average
                soft_out_ave_seq = torch.cat(soft_out_ave_seq,dim=2)



                summed_soft_out = soft_out_seq_large+soft_out_seq_middle+soft_out_seq_small+soft_out_seq_below_large+soft_out_seq_extra_middle
                soft_out_ensem_zeros = torch.zeros_like(soft_label_seq)
                soft_out_ensem_ones = torch.ones_like(soft_label_seq)
                comb_soft_out = torch.cat([soft_out_seq_large,soft_out_seq_middle,soft_out_seq_small,soft_out_seq_below_large,soft_out_seq_extra_middle],dim=0)

                comb_soft_out_std = torch.std(comb_soft_out,dim=0,unbiased=True,keepdim=True)
                comb_soft_out_mean = torch.mean(comb_soft_out,dim=0,keepdim=True)
                uncertainty_map = comb_soft_out_std/comb_soft_out_mean


                soft_out_ensem_mid = torch.where(summed_soft_out>=3,soft_out_ensem_ones,soft_out_ensem_zeros)

                uncertainty_map = uncertainty_map[0,1,:,:,:].squeeze().cpu().numpy()
                label_map = soft_label_seq[0,1,:,:,:].squeeze().cpu().numpy()
                prediction_mid_map = soft_out_ensem_mid[0,1,:,:,:].squeeze().cpu().numpy()
                input_map = input_slice_seq[0,0,:,:,:].squeeze().cpu().numpy()
                
                data_dict = {'uncertanty_map':uncertainty_map,'label_map':label_map,'pred_map':prediction_mid_map,'input_map':input_map}
                np.savez(out_npz_path,data_dict)



                # gtv_dice_small = get_classwise_dice(soft_out_seq_small,soft_label_seq).cpu().numpy()
                # gtv_dice_middle = get_classwise_dice(soft_out_seq_middle,soft_label_seq).cpu().numpy()
                # gtv_dice_large = get_classwise_dice(soft_out_seq_large,soft_label_seq).cpu().numpy()
                # gtv_dice_extra_middle = get_classwise_dice(soft_out_seq_extra_middle,soft_label_seq).cpu().numpy()
                # gtv_dice_below_large = get_classwise_dice(soft_out_seq_below_large,soft_label_seq).cpu().numpy()

                # gtv_dice_ensem_mid = get_classwise_dice(soft_out_ensem_mid,soft_label_seq).cpu().numpy()
                
                # gtv_dice_out_ave = get_classwise_dice(soft_out_ave_seq,soft_label_seq).cpu().numpy()

                # avg_test_dsc_small.append(gtv_dice_small)
                # avg_test_dsc_middle.append(gtv_dice_middle)
                # avg_test_dsc_large.append(gtv_dice_large)
                # avg_test_dsc_extra_middle.append(gtv_dice_extra_middle)
                # avg_test_dsc_below_large.append(gtv_dice_below_large)

                # avg_test_dsc_ensem_mid.append(gtv_dice_ensem_mid)
                # avg_test_dsc_out_ave.append(gtv_dice_out_ave)


                # if len(soft_out_seq) == 0:
                #     print('\n\n\n\n')
                #     print("$"*10)
                #     print("No prediction with --> {}".format(names))
                #     print("Present Classes --> {}".format(present_classes))
                #     print("$"*10)
                #     print('\n\n\n\n')
                #     continue

                print("Result in iter--{} for:{}, with Pixel-spacing:{}".format(iter_test,names,pixel_spacing))
                print("Result in iter--{} for:{}, with Pixel-spacing:{}".format(iter_test,names,pixel_spacing),file=infer_txt)
                print('Soft-out-shape',soft_out_seq_small.shape,'Soft Label-seq',soft_label_seq.shape)
                print('\n\n***************')
                # for c in range(1,class_num):
                #     print('Small-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_small[c]))
                #     print('Small-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_small[c]),file=infer_txt)
                #     print('Middle-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_middle[c]))
                #     print('Middle-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_middle[c]),file=infer_txt)
                #     print('Large-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_large[c]))
                #     print('Large-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_large[c]),file=infer_txt)
                #     print('Below-Large-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_below_large[c]))
                #     print('Below-Large-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_below_large[c]),file=infer_txt)
                #     print('Extra-Middle-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_extra_middle[c]))
                #     print('Extra-Middle-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_extra_middle[c]),file=infer_txt)
                #     print('\n****************** Ensemble Results ******************')
                #     print('Ensemble Majority-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_ensem_mid[c]))
                #     print('Ensemble Majority-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_ensem_mid[c]),file=infer_txt)
                #     print('Aggregated Output-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_out_ave[c]))
                #     print('Aggregated Output-class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice_out_ave[c]),file=infer_txt)


                print('Done Testing for iter-->{}'.format(iter_test))
                print('Done Testing for iter-->{}'.format(iter_test),file=infer_txt)
                print('\n\n\n')
                print('\n\n\n',file=infer_txt)


        print("\n\n\n")
        print("\n\n\n",file=infer_txt)
        print("*****"*10)
        print("*****"*10,file=infer_txt)

        # avg_time = (time.time() - start_time) / len(self.test_loder)
        # avg_test_dsc_small = np.asarray(avg_test_dsc_small).mean(axis = 0)
        # avg_test_dsc_middle = np.asarray(avg_test_dsc_middle).mean(axis = 0)
        # avg_test_dsc_large = np.asarray(avg_test_dsc_large).mean(axis = 0)
        # avg_test_dsc_extra_middle = np.asarray(avg_test_dsc_extra_middle).mean(axis = 0)
        # avg_test_dsc_below_large = np.asarray(avg_test_dsc_below_large).mean(axis = 0)

        # avg_test_dsc_ensem_mid = np.asarray(avg_test_dsc_ensem_mid).mean(axis = 0)
        # avg_test_dsc_out_ave = np.asarray(avg_test_dsc_out_ave).mean(axis=0)

        # print("average testing time {0:}".format(avg_time))
        # print("Average DSC result for total iter--{}".format(iter_test+1))
        # for c in range(1,class_num):
        #     print('Small--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_small[c]))
        #     print('Small--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_small[c]),file=infer_txt)            
        #     print('Middle--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_middle[c]))
        #     print('Middle--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_middle[c]),file=infer_txt)            
        #     print('Large--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_large[c]))
        #     print('Large--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_large[c]),file=infer_txt)            
        #     print('Extra-Middle--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_extra_middle[c]))
        #     print('Extra-Middle--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_extra_middle[c]),file=infer_txt)            
        #     print('Below-Large--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_below_large[c]))
        #     print('Below-Large--Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_below_large[c]),file=infer_txt)            

        #     print('\n******************Average Ensemble Results ******************')
        #     print('Ave-Ensemble Majority-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_ensem_mid[c]))
        #     print('Ave-Ensemble Majority-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_ensem_mid[c]),file=infer_txt)
        #     print('Ave-Aggregated Output-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_out_ave[c]))
        #     print('Ave-Aggregated Output-class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc_out_ave[c]),file=infer_txt)


        print('*****************************************')
        print("Done Overall Testing")
        print('*****************************************\n\n\n')
        print('*****************************************',file=infer_txt)
        print("Done Overall Testing",file=infer_txt)
        print('*****************************************\n\n\n',file=infer_txt)


    def run(self,exp):
        agent.__create_dataset()
        agent.__create_network()
        if(self.stage == 'dsc'):
            self.__dsc_eval(exp)
        elif self.stage=='surface_area':
            self.__surface_dist_eval(exp)
        elif self.stage == 'sensi_spec_ppv':
            self.__sensi_speci_ppv(exp)
        elif self.stage == 'uncertainity':
            self.__uncertainty_estimation(exp)

'''
**************************************************
average testing time 5.238658553675601
Average DSC result for total iter--19
Small--Average--class_1_dice,Test_dice_value:0.7732396595513213
Middle--Average--class_1_dice,Test_dice_value:0.7720220492940687
Large--Average--class_1_dice,Test_dice_value:0.7735499341332676

******************Average Ensemble Results ******************
Ave-Ensemble High-Confidence-class_1_dice,Test_dice_value:0.7614137357925139
Ave-Ensemble Mid-Confidence-class_1_dice,Test_dice_value:0.7728409838015808
Ave-Ensemble Low-Confidence-class_1_dice,Test_dice_value:0.7839031834239033
*****************************************
Done Overall Testing
*****************************************
'''

#New Set of Runs
'''
**************************************************
average testing time 2.8622193587453744
Average DSC result for total iter--19
Small--Average--class_1_dice,Test_dice_value:0.7732396595513213
Middle--Average--class_1_dice,Test_dice_value:0.7720220492940687
Large--Average--class_1_dice,Test_dice_value:0.7735499341332676
Extra-Middle--Average--class_1_dice,Test_dice_value:0.7862401045214049
Below-Large--Average--class_1_dice,Test_dice_value:0.7656907729897495

******************Average Ensemble Results ******************
Ave-Ensemble High-Confidence-class_1_dice,Test_dice_value:0.7489827726711004
Ave-Ensemble Mid-Confidence-class_1_dice,Test_dice_value:0.7909192124756865
Ave-Ensemble Low-High-Confidence-class_1_dice,Test_dice_value:0.7820826709833707
Ave-Ensemble Low-Mid-Confidence-class_1_dice,Test_dice_value:0.7881185768176346
Ave-Ensemble Low-Confidence-class_1_dice,Test_dice_value:0.759529936281834
*****************************************
Done Overall Testing
*****************************************

HC Average-Excess:0.0818, Average-Deficient:0.3532
Mid Average-Excess:0.2274, Average-Deficient:0.2209
Low HC Average-Excess:0.1388, Average-Deficient:0.2748
Low Mid Average-Excess:0.4127, Average-Deficient:0.171
LC Average-Excess:0.8509, Average-Deficient:0.1209

'''

import random
import numpy as np
seed = 30121994
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
# torch.set_deterministic(True)

if __name__ == "__main__":
    if(len(sys.argv) < 4):
        print('Number of arguments should be 3. e.g.')
        print(sys.argv)
        print('python train_infer.py train config.cfg exp')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    exp = str(sys.argv[3])



    config   = parse_config(cfg_file)
    agent    = TrainInferAgent(config, stage,exp)
    agent.run(exp)

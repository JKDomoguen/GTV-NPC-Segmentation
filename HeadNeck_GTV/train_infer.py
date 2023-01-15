# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
import time
import pandas as pd
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
import copy
import nibabel

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
from pymic.io.nifty_dataset import NiftyDataset2
from pymic.io.transform3d import get_transform
from pymic.train_infer.net_factory import get_network
from pymic.train_infer.loss import *
from pymic.train_infer.get_optimizer import get_optimiser
from pymic.util.image_process import convert_label
from pymic.util.parse_config import parse_config
from pymic.net3d.unet_3D_dv_semi import unet_3D_dv_semi
from pymic.net3d.model import UNet3D,ResidualUNet3D
import matplotlib.pyplot as plt
from PIL import Image

from pymic.self_supervised_tasks.algorithms.relative_patch_location import RelativePatchLocationModel
from pymic.self_supervised_tasks.algorithms.rotation import RotationModel
from pymic.self_supervised_tasks.algorithms.matching_net import MatchingModel
from pymic.self_supervised_tasks.algorithms.rpl_rot_exemp import RPL_ROT_EXEMP_MODEL


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

def get_train_test_paths(
    train_csv_path:str,
    test_csv_path:str,
    train_root_dir:str,
    test_root_dir:str,
    ):
    """_summary_

    Args:
        train_csv_path (str): train path must be in csv
        test_csv_path (str): 
        train_root_dir (str): root directory where the NPC are found
        test_root_dir (str): 

    Returns:
        _type_: tuple
    """
    train_paths_df = pd.read_csv(train_csv_path)
    test_paths_df = pd.read_csv(test_csv_path)
    train_img_paths,train_mask_paths = [],[]
    test_img_paths,test_mask_paths = [],[]

    for index,row in train_paths_df.iterrows():
        img_mask_paths = row.tolist()
        train_img_paths.append(os.path.join(train_root_dir,img_mask_paths[0]))
        train_mask_paths.append(os.path.join(train_root_dir,img_mask_paths[1]))

    for index,row in test_paths_df.iterrows():
        img_mask_paths = row.tolist()
        test_img_paths.append(os.path.join(test_root_dir,img_mask_paths[0]))
        test_mask_paths.append(os.path.join(test_root_dir,img_mask_paths[1]))
    return train_img_paths,train_mask_paths,test_img_paths,test_mask_paths

NORMAL_TRAIN_DIR = "/media/disk1/jansen/code_rad/Dataset_Rad2/gtv_normal_processed_uni/small_scale"

class TrainInferAgent():
    def __init__(self, 
                 config, 
                 data_fold=None,
                 data_scale=None,
                 stage = 'train',
                 exp='debug'
                 ):
        self.config = config
        self.stage  = stage
        self.data_fold = data_fold
        self.data_scale = data_scale

        if stage == 'train':            
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
            self.model_name = self.config['network']['net_type'].lower()
            chpt_prefx  = os.path.join(self.config['training']['checkpoint_prefix'],self.model_name)
            self.output_dir = f"{chpt_prefx}__{dt_string}"
        elif stage == 'test' or stage =='infer':
            now = datetime.now()
            dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
            model_name = self.config['testing']['checkpoint_name'].split('/')[-1].strip('.pt')
            folder_name = f"{model_name}__{dt_string}"
            result_dir = self.config['testing']['output_dir']
            self.output_dir = os.path.join(result_dir,folder_name)
        self.exp = exp
        if exp != 'debug':        
            os.makedirs(self.output_dir,exist_ok=True)

        assert(stage in ['train', 'inference', 'test','ensemble'])

    def __create_dataset(self,test_dataset):
        
        if self.data_fold == None:
            self.data_fold = self.config['dataset']['data_fold']
        if self.data_scale == None:
            self.data_scale = self.config['dataset']['data_scale']

        data_root_dir = self.config['dataset']['data_root_dir']
        config_csv_dir = self.config['dataset']['config_csv_dir']
        
        if self.data_scale == 'small':
            data_scale_dir = "small_scale"
        elif self.data_scale == 'medium':
            data_scale_dir = "middle_scale"
        elif self.data_scale == 'extra_medium':
            data_scale_dir = "extra_middle_scale"  
        elif self.data_scale == 'below_large':
            data_scale_dir = "below_large_scale"
        elif self.data_scale == 'large':
            data_scale_dir = "large_scale"
        
        root_dir_train = os.path.join(data_root_dir,f"iteration_{self.data_fold}/gtv_train/{data_scale_dir}")
        root_dir_test = os.path.join(data_root_dir,f"iteration_{self.data_fold}/gtv_test/uncut_scale")
        
        train_csv_path = os.path.join(config_csv_dir,f'train_pathts_iter-{self.data_fold}.csv')
        test_csv_path = os.path.join(config_csv_dir,f'test_pathts_iter-{self.data_fold}.csv')
        train_img_paths,train_mask_paths,test_img_paths,test_mask_paths = get_train_test_paths(
                                                                            train_csv_path,
                                                                            test_csv_path,
                                                                            root_dir_train,
                                                                            root_dir_test
                                                                            )
        weight_training = self.config['dataset']['weight_training']
        num_train_samples = int(len(train_img_paths)*weight_training)
        rand_idx = random.sample(range(len(train_img_paths)), num_train_samples)
        train_img_paths = [train_img_paths[idx] for idx in rand_idx]
        train_mask_paths = [train_mask_paths[idx] for idx in rand_idx]

        


        modal_num = self.config['dataset']['modal_num']

        if(self.stage == 'train'):
            transform_names = self.config['dataset']['train_transform']
            validtransform_names = self.config['dataset']['valid_transform']
            self.validtransform_list = [get_transform(name, self.config['dataset']) \
            for name in validtransform_names if name != 'RegionSwop']
        else:
            transform_names = self.config['dataset']['test_transform']
        self.transform_list = [get_transform(name, self.config['dataset']) \
            for name in transform_names if name != 'RegionSwop']

        if('RegionSwop' in transform_names):
            self.region_swop = get_transform('RegionSwop', self.config['dataset'])
        else:
            self.region_swop = None

        if(self.stage == 'train'):
            train_dataset = NiftyDataset2(img_paths=train_img_paths,
                                label_paths  = train_mask_paths,
                                modal_num = modal_num,
                                with_label= True,
                                transform = transforms.Compose(self.transform_list))

            valid_dataset = NiftyDataset2(img_paths=test_img_paths,
                                label_paths  = test_mask_paths,
                                modal_num = modal_num,
                                with_label= True,
                                transform = transforms.Compose(self.validtransform_list))

            batch_size = self.config['training']['batch_size']

            self.train_loader = torch.utils.data.DataLoader(train_dataset,pin_memory=True,
                drop_last=True , batch_size = batch_size, shuffle=True, num_workers=8)

            self.valid_loader = torch.utils.data.DataLoader(valid_dataset,
                batch_size = 1, shuffle=False, num_workers=2)

        else:
            test_dataset = NiftyDataset(root_dir=root_dir_test,
                                csv_file  = test_csv,
                                modal_num = modal_num,
                                with_label= True,
                                transform = transforms.Compose(self.transform_list))

            batch_size = 1
            self.test_loder = torch.utils.data.DataLoader(test_dataset,
                batch_size=batch_size, shuffle=False, num_workers=batch_size)
            

    def __create_network(self):
        
        self.net = get_network(self.config['network'])
        network_dict = self.net.state_dict()
        self.__load_encoder_from_ssl_model()

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

    def __load_encoder_from_ssl_model(self):

        if self.config['SSL']['algorithm'] == 'rpl':        
            ssl_model = RelativePatchLocationModel(self.config)
        elif self.config['SSL']['algorithm'] == 'rotation':        
            ssl_model = RotationModel(self.config)
        elif self.config['SSL']['algorithm'] == 'matching':        
            ssl_model = MatchingModel(self.config)
        elif self.config['SSL']['algorithm'] == 'rpl-rot-exemp':
            ssl_model = RPL_ROT_EXEMP_MODEL(self.config)
        else:
            ssl_model = RelativePatchLocationModel(self.config)


        if self.config['finetuning']['ssl_trained_model'] == None:
            print("Not using SSL-Trained model as pretrained model")
            print("Using Raw model")
            return

        if os.path.isfile(self.config['finetuning']['ssl_trained_model']):            
            ssl_model_ckpt = torch.load(self.config['finetuning']['ssl_trained_model'],map_location=self.config['training']['device_name'])
            # ssl_model_ckpt = torch.load(self.config['finetuning']['ssl_trained_model'])
            ssl_model.load_state_dict(ssl_model_ckpt['model_state_dict'])
            print("Using Pretrained Model At: {}".format(self.config['finetuning']['ssl_trained_model']))
        else:
            raise("No Pre-Trained SSL Model")


        ssl_model_dict = ssl_model._encoder.state_dict()
        net_model_dict = self.net.state_dict()
        ssl_model_dict = {k: v for k, v in ssl_model_dict.items() if k in net_model_dict}

        
        net_model_dict.update(ssl_model_dict)
        self.net.load_state_dict(net_model_dict)
        # print("Special Training for Raw Model with Encoder Frozen")
        self.net.freeze_encoder(ssl_model_dict.keys())
        print("Succesfully Loaded: {}".format(self.config['finetuning']['ssl_trained_model'].split('model/')[-1]))    
        print("List of Traininable Parameters")
        print("\n\n**************************")
        num_param_grad = 0
        num_param_w0_grad = 0
        for (name,param) in self.net.named_parameters():
            num_param = param.numel()
            if param.requires_grad:
                print("Requires Gradient, Parameter:{}, Num-Parameter:{}".format(name,num_param))
                num_param_grad += num_param
            else:
                print("Does Not Requires Gradient, Parameter:{}, Num-Parameter:{}".format(name,num_param))
                num_param_w0_grad += num_param
        
        print("Number of parameters for Trainable Layers:{}".format(num_param_grad))
        print("Number of paramaters for Non-Trainable Layers:{}".format(num_param_w0_grad))

        print("Encoder Network is Also Frozen")
        print("**************************\n\n")

    def __create_optimizer(self):
        self.optimizer = get_optimiser(self.config['training']['optimizer'],
                self.net.parameters(),
                self.config['training'])
        last_iter = -1
        self.schedule = optim.lr_scheduler.MultiStepLR(self.optimizer,
                self.config['training']['lr_milestones'],
                self.config['training']['lr_gamma'],
                last_epoch = last_iter)

    def __train(self,exp):
        if self.exp != 'debug':
            self.train_txt = open( os.path.join(self.output_dir,'output_{}.txt'.format(exp)) ,'a')
        else:
            self.train_txt = open('output_{}.txt'.format(exp) ,'w')
        # self.train_txt = open('output_{}.txt'.format(exp),'a')

        print(self.net,file=self.train_txt)
        print("Total number of parameters:{}".format(count_parameters(self.net)),file=self.train_txt)
        print(self.net)
        print("Total number of parameters:{}".format(count_parameters(self.net)))

        device = torch.device(self.config['training']['device_name'])
        self.net.to(device)
        VAL_SLICE = 16
        VAL_EPOCH = 50
        epoch_save = 10000

        # summ_writer = SummaryWriter(self.config['training']['summary_dir'])
        summ_writer = SummaryWriter(self.output_dir)
        chpt_prefx  = self.config['training']['checkpoint_prefix']
        loss_func   = self.config['training']['loss_function']
        iter_start  = self.config['training']['iter_start']
        class_num   = self.config['network']['class_num']
        loss_obj = SegmentationLossCalculator(loss_func)

        if(iter_start > 0):
            checkpoint_file = "{0:}_{1:}.pt".format(chpt_prefx, iter_start)
            self.checkpoint = torch.load(checkpoint_file)
            assert(self.checkpoint['iteration'] == iter_start)
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            self.checkpoint = None

        self.__create_optimizer()
        
        with open(os.path.join(self.output_dir,'config_file.txt'),'w') as f:
            print("Config File",file=f)
            for section in self.config:
                print("\n\nKey-Val pairs under Section -----> {}".format(section),file=f)
                for key,val in self.config[section].items():
                    print("{} ------> {}".format(key,val),file=f)

        print("{0:} training start".format(str(datetime.now())[:-7]))
        print("{0:} training start".format(str(datetime.now())[:-7]),file=self.train_txt)
        
        
        tqdm_gem = tqdm.tqdm(self.train_loader)
        # total_epoch = 1+iter_max//len(self.train_loader)
        total_epoch = self.config['training']['max_epoch']
        it = 0
        self.net.train()
        epoch = 0
        self._test_epoch(epoch)
        while epoch < total_epoch:
            train_loss      = 0
            train_dice_list = []
            for tr_idx,data in enumerate(tqdm_gem):
            # for tr_idx,data in enumerate(self.train_loader):
                # get the inputs
                inputs, labels = data['image'], data['label']
                labels = torch.where(labels != 1,0,1)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print('**********Unique Values*****************',torch.unique(torch.flatten(labels)))
                train_dice_per_patient = []
                train_loss_per_patient = []

                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                soft_y  = get_soft_label(labels, class_num,device)
                ce_loss = get_ce_loss(outputs,labels)
                dice_loss = loss_obj.get_loss(outputs, soft_y) 
                loss    = (dice_loss + ce_loss)/2

                loss.backward()
                self.optimizer.step()
                it += 1

                if len(torch.unique(labels))==class_num:
                    train_loss_per_patient.append(loss.detach().item())
                    outputs_argmax = torch.argmax(outputs.detach(), dim = 1, keepdim = True)
                    soft_out  = get_soft_label(outputs_argmax, class_num,device)
                    dice_list = get_classwise_dice(soft_out, soft_y)
                    train_dice_per_patient.append(dice_list.detach().cpu().numpy())
                    del outputs_argmax

                del inputs
                del labels
                del outputs
                del loss
                # torch.cuda.empty_cache()

                if len(train_loss_per_patient) == 0:
                    continue

                train_dice_per_patient = np.asarray(train_dice_per_patient).mean(axis = 0)
                train_loss_per_patient = sum(train_loss_per_patient) /len(train_loss_per_patient)
                print("Epoch-->{}, Iteration-->{}, DSC-Val-->{},  Loss-Value--->{}".format(epoch,it,train_dice_per_patient,train_loss_per_patient))
                print("Epoch-->{}, Iteration-->{}, DSC-Val-->{},  Loss-Value--->{}".format(epoch,it,train_dice_per_patient,train_loss_per_patient),file=self.train_txt)

                # evaluate performance on validation set
                train_loss += train_loss_per_patient
                train_dice_list.append(train_dice_per_patient)
                # break

            if len(train_dice_list) == 0:
                print("Zero in Loader")
                continue            

            train_avg_loss = train_loss / len(train_dice_list)
            train_cls_dice = np.asarray(train_dice_list).mean(axis = 0)
            train_avg_dice = train_cls_dice.mean()
            run_avg_loss = running_mean(train_avg_loss,N=20)

            print('\n\n')
            print('\n\n',file=self.train_txt)
            print('****'*8)
            print('****'*8,file=self.train_txt)

            print("{0:} it {1:}, loss {2:.4f}".format(
                str(datetime.now())[:-7], it + 1, train_avg_loss))
            print("{0:} it {1:}, loss {2:.4f}".format(
                str(datetime.now())[:-7], it + 1, train_avg_loss),file=self.train_txt)
            print("Running-Average Loss:{}".format(run_avg_loss))
            print("Running-Average Loss:{}".format(run_avg_loss),file=self.train_txt)
            print('train cls dice', train_cls_dice.shape, train_cls_dice)
            print('train cls dice', train_cls_dice.shape, train_cls_dice,file=self.train_txt)
            for c in range(class_num):
                print('class_{}_dice, train_dice_value:{}, iteration-{}'.format(c,train_cls_dice[c],it+1))
                print('class_{}_dice, train_dice_value:{}, iteration-{}'.format(c,train_cls_dice[c],it+1),file=self.train_txt)
            print('****'*8)
            print('****'*8,file=self.train_txt)
            print('\n\n')
            print('\n\n',file=self.train_txt)
            epoch += 1
            self.schedule.step()
            tqdm_gem = tqdm.tqdm(self.train_loader)

            if (epoch % VAL_EPOCH == 0):
                avg_test_gtv_dsc = self._test_epoch(epoch)

            if (epoch % epoch_save ==  0):
                print('\n\n\n\n\n')
                print('\n\n\n\n\n',file=self.train_txt)
                print("####"*5)
                print("####"*5,file=self.train_txt)
                print("Saving Model")
                print("Saving Model",file=self.train_txt)

                save_dict = {'iteration': it + 1,
                             'model_state_dict': self.net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict()}
                # save_name = "{0:}_{1:}.pt".format(self.model_name, it + 1)
                save_name = "{0:}_{1:}.pt".format(self.model_name, epoch)
                save_path = os.path.join(self.output_dir,save_name)
                torch.save(save_dict, save_path)
                print("Full Save Path:{}".format(save_path))
                print("Full Save Path:{}".format(save_path),file=self.train_txt)
                print("Saved")
                print("Saved",file=self.train_txt)
                print("####"*5)
                print("####"*5,file=self.train_txt)
                print('\n\n\n\n\n')
                print('\n\n\n\n\n',file=self.train_txt)
        summ_writer.close()

    def _test_epoch(self,epoch):

        torch.cuda.empty_cache()
        print('\n\n\n\n\n')
        print('\n\n\n\n\n',file=self.train_txt)
        print("$$$$"*5)
        print("$$$$"*5,file=self.train_txt)
        # self.net.eval()
        print("Begin Test-Validation at Epoch-->{}".format(epoch))
        print("Begin Test at Epoch-->{}".format(epoch),file=self.train_txt)

        device = torch.device(self.config['training']['device_name'])
        self.net.eval()
        # output_dir   = self.config['testing']['output_dir']
        class_num    = self.config['network']['class_num']
        SLICE = 16

        start_time = time.time()
        avg_test_dsc = []
        with torch.no_grad():
            for iter_test,data in enumerate(self.valid_loader):
                images,labels = data['image'],data['label']
                #Do This for UGenesis Only
                images,labels = images[:,:,:,:384,:384],labels[:,:,:,:384,:384]
                present_classes = torch.unique(labels.flatten()).numpy()
                labels = torch.where(labels != 1,0,1)
                images, labels = images.to(device), labels.to(device)
                total_gtv_voxels = torch.sum(labels.flatten()).item() 
                names  = data['names']
                pixel_spacing = data['spacing']
                print("Processing --> {} with input and mask shape --> {}, {} at epoch--{}".format(names,images.shape,labels.shape,epoch))
                print("Processing --> {} with input and mask shape --> {}, {} at epoch--{}".format(names,images.shape,labels.shape,epoch),file=self.train_txt)

                needed_num_slice = SLICE - int(images.shape[2])%SLICE

                last_inp_slice,last_label_slice = images[0,0,-1,:,:],labels[0,0,-1,:,:]

                inp_slice = last_inp_slice.repeat(1,1,needed_num_slice,1,1)
                label_slice_extra = last_label_slice.repeat(1,1,needed_num_slice,1,1)

                images = torch.cat((images,inp_slice),dim=2)
                labels = torch.cat((labels,label_slice_extra),dim=2)

                images = torch.split(images,SLICE,2)
                labels = torch.split(labels,SLICE,2)

                soft_out_seq = []
                soft_label_seq = []

                for idx,(inputs_slice,labels_slice) in enumerate(zip(images,labels)):

                    if len(torch.unique(labels_slice))<class_num:
                        continue

                    outputs = self.net(inputs_slice)
                    print('**********Unique Values*****************',torch.unique(torch.flatten(labels_slice)))
                    soft_y = get_soft_label(labels_slice,class_num,device)
                    outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
                    
                    total_slice_gtv_voxels = torch.sum(labels_slice.flatten()).cpu().item()
                    ratio_slice_gtv_voxels = round(total_slice_gtv_voxels/total_gtv_voxels,4)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax = outputs_argmax[:,:,:end_slice,:,:]
                        soft_y = soft_y[:,:,:end_slice,:,:]


                    soft_out  = get_soft_label(outputs_argmax, class_num,device)


                    soft_out_seq.append(soft_out)
                    soft_label_seq.append(soft_y)
                    dsc_value_iter = get_classwise_dice(soft_out, soft_y).cpu().numpy()
                    print("DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels))
                    print("DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels),file=self.train_txt)

                    del outputs
                    del outputs_argmax
                    del inputs_slice
                    del labels_slice
                    torch.cuda.empty_cache()

                if len(soft_out_seq) == 0:
                    print('\n\n\n\n')
                    print("$"*10)
                    print("No prediction with --> {}".format(names))
                    print("Present Classes --> {}".format(present_classes))
                    print("$"*10)
                    print('\n\n\n\n')
                    continue
                
                soft_label_seq = torch.cat(soft_label_seq,dim=2)
                soft_out_seq = torch.cat(soft_out_seq,dim=2)
                print(soft_label_seq.shape,soft_out_seq.shape,'sequence')
                gtv_dice = get_classwise_dice(soft_out_seq,soft_label_seq).cpu().numpy()
                gtv_dice_gt = get_classwise_dice(soft_out_seq,soft_out_seq).cpu().numpy()
                

                gtv_dice_gt2 = get_classwise_dice(soft_label_seq,soft_label_seq).cpu().numpy()
                # print("Sanity Check1:{} and Sanity Check2:{}".format(len(gtv_dice_gt),len(gtv_dice_gt2) ))
                print("Sanity Check1:{} and Sanity Check2:{}".format(gtv_dice_gt,gtv_dice_gt2 ))
                avg_test_dsc.append(gtv_dice)


                print("Epoch--->{}".format(epoch))
                print("Epoch--->{}".format(epoch),file=self.train_txt)
                print("Result in iter--{} for:{}, with Pixel-spacing:{}".format(iter_test,names,pixel_spacing))
                print("Result in iter--{} for:{}, with Pixel-spacing:{}".format(iter_test,names,pixel_spacing),file=self.train_txt)
                print('Soft-out-shape',soft_out_seq.shape,'Soft Label-seq',soft_label_seq.shape)
                for c in range(class_num):
                    print('class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice[c]))
                    print('class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice[c]),file=self.train_txt)
                print('Done Testing for iter-->{}'.format(iter_test))
                print('Done Testing for iter-->{}'.format(iter_test),file=self.train_txt)
                print('\n\n\n')
                print('\n\n\n',file=self.train_txt)
                torch.cuda.empty_cache()


        print("\n\n\n")
        print("\n\n\n",file=self.train_txt)
        print("*****"*10)
        print("*****"*10,file=self.train_txt)

        avg_time = (time.time() - start_time) / len(self.valid_loader)
        avg_test_dsc = np.asarray(avg_test_dsc).mean(axis = 0)
        print("Final Test Result at Epoch-->{}".format(epoch))
        print("Average testing time {0:}".format(avg_time))
        print("Average DSC result for total iter--{}".format(iter_test+1))

        print("Final Test Result at Epoch-->{}".format(epoch),file=self.train_txt)        
        print("Average testing time {0:}".format(avg_time),file=self.train_txt)
        print("Average DSC result for total iter--{}".format(iter_test+1),file=self.train_txt)

        for c in range(0,class_num):
            print('Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc[c]))
            print('Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc[c]),file=self.train_txt)            

        print('*****************************************')
        print("Done Testing")
        print('*****************************************\n\n\n')


        print('*****************************************',file=self.train_txt)
        print("Done Testing",file=self.train_txt)
        print('*****************************************\n\n\n',file=self.train_txt)

        self.net.train()
        avg_gtv_dsc_test = avg_test_dsc[0]
        return avg_gtv_dsc_test

    def __infer(self,exp,checkpoint):
        if self.exp != 'debug':
            infer_txt = open( os.path.join(self.output_dir,'infer_output_{}.txt'.format(exp)) ,'a')
        else:
            infer_txt = open('infer_output_{}.txt'.format(exp) ,'w')
        device = torch.device(self.config['training']['device_name'])
        self.net.to(device)
        # laod network parameters and set the network as evaluation mode
        if checkpoint == None:            
            self.checkpoint = torch.load(self.config['testing']['checkpoint_name'])
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            # ckpt_dir = os.path.join(self.config['testing']['checkpoint_dir'],checkpoint)
            self.checkpoint = torch.load(checkpoint)
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
        # self.net.train()
        print('Starting to test using checkpoint:{}.pt'.format(checkpoint))
        print('Starting to test using checkpoint:{}.pt'.format(checkpoint),file=infer_txt)
        self.net.eval()

        # output_dir   = self.config['testing']['output_dir']
        class_num    = self.config['network']['class_num']
        SLICE = 8

        start_time = time.time()
        avg_test_dsc = []
        with torch.no_grad():
            for iter_test,data in enumerate(self.test_loder):
                labels_hist = np.histogram(data['label'].numpy().flatten())
                images,labels = data['image'],data['label']
                present_classes = torch.unique(labels.flatten()).numpy()
                images, labels = images.to(device), labels.to(device)
                #Get total-voxels of GTV
                total_gtv_voxels = torch.sum(labels.flatten()).item() 

                names  = data['names']
                pixel_spacing = data['spacing']
                print("Processing --> {} with input and mask shape --> {}, {}".format(names,images.shape,labels.shape))
                print("Processing --> {} with input and mask shape --> {}, {}".format(names,images.shape,labels.shape),file=infer_txt)
                needed_num_slice = SLICE - int(images.shape[2])%SLICE
                last_inp_slice,last_label_slice = images[0,0,-1,:,:],labels[0,0,-1,:,:]
                inp_slice = last_inp_slice.repeat(1,1,needed_num_slice,1,1)
                label_slice_extra = last_label_slice.repeat(1,1,needed_num_slice,1,1)
                images = torch.cat((images,inp_slice),dim=2)
                labels = torch.cat((labels,label_slice_extra),dim=2)
                images = torch.split(images,SLICE,2)
                labels = torch.split(labels,SLICE,2)
                temp_output_list = []
                valid_dice_list = []
                soft_out_seq = []
                for idx,(inputs_slice,labels_slice) in enumerate(zip(images,labels)):

                    if len(np.unique(labels_slice.cpu().numpy().flatten())) < class_num:
                        continue
                    outputs = self.net(inputs_slice)
                    # print(f"Inputs-shape ---->{inputs_slice.shape} ***** Mask-Shape --->{labels_slice.shape} **** Output-Shape ---->{outputs.shape}")
                    soft_y = get_soft_label(labels_slice,class_num,device)
                    outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax = outputs_argmax[:,:,:end_slice,:,:]
                        soft_y = soft_y[:,:,:end_slice,:,:]

                    temp_output_list.append(outputs_argmax.cpu().numpy().astype(np.uint8))
                    soft_out  = get_soft_label(outputs_argmax, class_num,device)
                    soft_out_seq.append(soft_out.cpu().numpy().astype(np.uint8).squeeze(axis=0))
                    dice_list = get_classwise_dice(soft_out, soft_y)
                    valid_dice_list.append(dice_list.cpu().numpy())

                    dsc_value_iter = dice_list.cpu().numpy()
                    total_slice_gtv_voxels = torch.sum(labels_slice.flatten()).item()
                    ratio_slice_gtv_voxels = round(total_slice_gtv_voxels/total_gtv_voxels,4)

                    print('\n')
                    print('******'*10)
                    print("DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels))
                    print("DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels),file=infer_txt)
                    print('******'*10)
                    print('\n')

                    del outputs
                    del outputs_argmax
                    del inputs_slice
                    del labels_slice
                    torch.cuda.empty_cache()
                if len(soft_out_seq) == 0:
                    print('\n\n\n\n')
                    print("$"*10)
                    print("No prediction with --> {}".format(names))
                    print("Present Classes --> {}".format(present_classes))
                    print("$"*10)
                    print('\n\n\n\n')
                    continue
                soft_out_seq = np.concatenate(soft_out_seq,axis=1)

                # print(np.unique(soft_out_seq[1].flatten()),'GTVp\n\n\n')                
                valid_cls_dice = np.asarray(valid_dice_list).mean(axis = 0)
                avg_test_dsc.append(valid_cls_dice)
                output_array = np.concatenate(temp_output_list,axis=2)
                # print(output_array.shape)
                print("\n\n\n")
                print("\n\n\n",file=infer_txt)
                print("*****"*10)
                print("*****"*10,file=infer_txt)
                print("Result in iter--{} for:{}, with Pixel-spacing:{}".format(iter_test,names,pixel_spacing))
                print("Result in iter--{} for:{}, with Pixel-spacing:{}".format(iter_test,names,pixel_spacing),file=infer_txt)
                print('Soft-out-shape',soft_out_seq.shape,'Output array-shape',output_array.shape)
                for c in range(class_num):
                    print('class_{}_dice,Test_dice_value:{}'.format(c,valid_cls_dice[c]))
                    print('class_{}_dice,Test_dice_value:{}'.format(c,valid_cls_dice[c]),file=infer_txt)
                print(f"Background-->{np.unique(soft_out_seq[0].flatten())}, GTV-->{np.unique(soft_out_seq[1].flatten())}")
                # output_hist = np.histogram(soft_out_seq[1].flatten())
                output_hist2 = np.histogram(output_array)
                print(f"Label ---> Bin:{labels_hist[0]}\nCount:{labels_hist[1]}\n")
                # print(f"Output ---> Bin:{output_hist[0]}\nCount:{output_hist[1]}\n")
                print(f"Output ---> Bin:{output_hist2[0]}\nCount:{output_hist2[1]}")
                print('\n\n\n')
                print('\n',file=infer_txt)

        avg_time = (time.time() - start_time) / len(self.test_loder)
        avg_test_dsc = np.asarray(avg_test_dsc).mean(axis = 0)
        print('Test result for checkpoint -- {}'.format(checkpoint))
        print("average testing time {0:}".format(avg_time))
        print("Average DSC result for total iter--{}".format(iter_test+1))
        
        print('Test result for checkpoint -- {}'.format(checkpoint),file=infer_txt)
        print("average testing time {0:}".format(avg_time),file=infer_txt)
        print("Average DSC result for total iter--{}".format(iter_test+1),file=infer_txt)
        for c in range(0,class_num):
            print('Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc[c]))
            print('Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc[c]),file=infer_txt)            
    

    def __infer2(self,exp,checkpoint):

        if self.exp != 'debug':
            infer_txt = open( os.path.join(self.output_dir,'infer_output_{}.txt'.format(exp)) ,'a')
        else:
            infer_txt = open('infer_output_{}.txt'.format(exp) ,'w')
        device = torch.device(self.config['training']['device_name'])
        self.net.to(device)
        # laod network parameters and set the network as evaluation mode
        if checkpoint == None:            
            self.checkpoint = torch.load(self.config['testing']['checkpoint_name'])
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            # ckpt_dir = os.path.join(self.config['testing']['checkpoint_dir'],checkpoint)
            self.checkpoint = torch.load(checkpoint)
            self.net.load_state_dict(self.checkpoint['model_state_dict'])

        print('Starting to test using checkpoint:{}.pt'.format(checkpoint))
        print('Starting to test using checkpoint:{}.pt'.format(checkpoint),file=infer_txt)
        self.net.eval()

        class_num    = self.config['network']['class_num']
        SLICE = 16

        start_time = time.time()
        avg_test_dsc = []
        with torch.no_grad():
            for iter_test,data in enumerate(self.test_loder):
                images,labels = data['image'],data['label']
                present_classes = torch.unique(labels.flatten()).numpy()
                images, labels = images.to(device), labels.to(device)
                total_gtv_voxels = torch.sum(labels.flatten()).item() 
                names  = data['names']
                pixel_spacing = data['spacing']
                print("Processing --> {} with input and mask shape --> {}, {} at Iteration--{}".format(names,images.shape,labels.shape,iter_test))
                print("Processing --> {} with input and mask shape --> {}, {} at Iteration--{}".format(names,images.shape,labels.shape,iter_test),file=infer_txt)
                needed_num_slice = SLICE - int(images.shape[2])%SLICE
                last_inp_slice,last_label_slice = images[0,0,-1,:,:],labels[0,0,-1,:,:]
                inp_slice = last_inp_slice.repeat(1,1,needed_num_slice,1,1)
                label_slice_extra = last_label_slice.repeat(1,1,needed_num_slice,1,1)
                images = torch.cat((images,inp_slice),dim=2)
                labels = torch.cat((labels,label_slice_extra),dim=2)
                images = torch.split(images,SLICE,2)
                labels = torch.split(labels,SLICE,2)

                soft_out_seq = []
                soft_label_seq = []

                for idx,(inputs_slice,labels_slice) in enumerate(zip(images,labels)):

                    if len(torch.unique(labels_slice))<class_num:
                        continue

                    outputs = self.net(inputs_slice)
                    soft_y = get_soft_label(labels_slice,class_num,device)
                    outputs_argmax = torch.argmax(outputs, dim = 1, keepdim = True)
                    
                    total_slice_gtv_voxels = torch.sum(labels_slice.flatten()).cpu().item()
                    ratio_slice_gtv_voxels = round(total_slice_gtv_voxels/total_gtv_voxels,4)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax = outputs_argmax[:,:,:end_slice,:,:]
                        soft_y = soft_y[:,:,:end_slice,:,:]


                    soft_out  = get_soft_label(outputs_argmax, class_num,device)
                    soft_out_seq.append(soft_out)
                    soft_label_seq.append(soft_y)
                    dsc_value_iter = get_classwise_dice(soft_out, soft_y).cpu().numpy()
                    print("DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels))
                    print("DSC-Value:{}, Slice-Voxels:{}, Total-Voxels:{}, Ratio-Slice-Voxels:{}".format(dsc_value_iter,total_slice_gtv_voxels,total_gtv_voxels,ratio_slice_gtv_voxels),file=infer_txt)

                    del outputs
                    del outputs_argmax
                    del inputs_slice
                    del labels_slice
                    torch.cuda.empty_cache()

                if len(soft_out_seq) == 0:
                    print('\n\n\n\n')
                    print("$"*10)
                    print("No prediction with --> {}".format(names))
                    print("Present Classes --> {}".format(present_classes))
                    print("$"*10)
                    print('\n\n\n\n')
                    continue
                
                soft_label_seq = torch.cat(soft_label_seq,dim=2)
                soft_out_seq = torch.cat(soft_out_seq,dim=2)
                gtv_dice = get_classwise_dice(soft_out_seq,soft_label_seq).cpu().numpy()
                avg_test_dsc.append(gtv_dice)


                print("Result in iter--{} for:{}, with Pixel-spacing:{}".format(iter_test,names,pixel_spacing))
                print("Result in iter--{} for:{}, with Pixel-spacing:{}".format(iter_test,names,pixel_spacing),file=infer_txt)
                print('Soft-out-shape',soft_out_seq.shape,'Soft Label-seq',soft_label_seq.shape)
                for c in range(class_num):
                    print('class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice[c]))
                    print('class_{}_dice,Test_dice_value:{}'.format(c,gtv_dice[c]),file=infer_txt)
                print('Done Testing for iter-->{}'.format(iter_test))
                print('Done Testing for iter-->{}'.format(iter_test),file=infer_txt)
                print('\n\n\n')
                print('\n\n\n',file=infer_txt)


        print("\n\n\n")
        print("\n\n\n",file=infer_txt)
        print("*****"*10)
        print("*****"*10,file=infer_txt)

        avg_time = (time.time() - start_time) / len(self.test_loder)
        avg_test_dsc = np.asarray(avg_test_dsc).mean(axis = 0)
        print('Test result for checkpoint -- {}'.format(checkpoint))
        print("average testing time {0:}".format(avg_time))
        print("Average DSC result for total iter--{}".format(iter_test+1))
        for c in range(0,class_num):
            print('Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc[c]))
            print('Average--class_{}_dice,Test_dice_value:{}'.format(c,avg_test_dsc[c]),file=infer_txt)            

        print('*****************************************')
        print("Done Overall Testing")
        print('*****************************************\n\n\n')
        print('*****************************************',file=infer_txt)
        print("Done Overall Testing",file=infer_txt)
        print('*****************************************\n\n\n',file=infer_txt)


    def __infer_ensemble(self,exp):

        if self.exp != 'debug':
            infer_txt = open( os.path.join(self.output_dir,'infer_output_{}.txt'.format(exp)) ,'a')
        else:
            infer_txt = open('infer_output_{}.txt'.format(exp) ,'w')
        device = torch.device(self.config['training']['device_name'])

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
        SLICE = 32

        start_time = time.time()
        avg_test_dsc_small = []
        avg_test_dsc_middle = []
        avg_test_dsc_large = []
        avg_test_dsc_extra_middle = []
        avg_test_dsc_below_large = []
        avg_test_dsc_ensem_mid = []
        avg_test_dsc_out_ave = []

        gtv_variance_np_dir = 'results/gtv_maps2'
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
                    soft_y = get_soft_label(labels_slice,class_num,device)
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


                    soft_out_small  = get_soft_label(outputs_argmax_small, class_num,device)
                    soft_out_seq_small.append(soft_out_small)
                    soft_label_seq.append(soft_y)                        
                    input_slice_seq.append(img_nifti_slice)

                    del outputs_argmax_small
                    
                    #Middle
                    output_middle = model_middle(inputs_slice)
                    outputs_argmax_middle = torch.argmax(output_middle, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_middle = outputs_argmax_middle[:,:,:end_slice,:,:]

                    soft_out_middle  = get_soft_label(outputs_argmax_middle, class_num,device)
                    soft_out_seq_middle.append(soft_out_middle)

                    del outputs_argmax_middle

                    #Large
                    output_large = model_large(inputs_slice)
                    outputs_argmax_large = torch.argmax(output_large, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_large = outputs_argmax_large[:,:,:end_slice,:,:]

                    soft_out_large  = get_soft_label(outputs_argmax_large, class_num,device)
                    soft_out_seq_large.append(soft_out_large)

                    del outputs_argmax_large

                    #Below Large
                    output_below_large = model_below_large(inputs_slice)
                    outputs_argmax_below_large = torch.argmax(output_below_large, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_below_large = outputs_argmax_below_large[:,:,:end_slice,:,:]

                    soft_out_below_large  = get_soft_label(outputs_argmax_below_large, class_num,device)
                    soft_out_seq_below_large.append(soft_out_below_large)
                    del outputs_argmax_below_large

                    #Extra Middle
                    output_extra_middle = model_extra_middle(inputs_slice)
                    outputs_argmax_extra_middle = torch.argmax(output_extra_middle, dim = 1, keepdim = True)

                    if idx == (len(images)-1):
                        end_slice = SLICE-needed_num_slice
                        outputs_argmax_extra_middle = outputs_argmax_extra_middle[:,:,:end_slice,:,:]

                    soft_out_extra_middle  = get_soft_label(outputs_argmax_extra_middle, class_num,device)
                    soft_out_seq_extra_middle.append(soft_out_extra_middle)
                    del outputs_argmax_extra_middle
                    
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
                    del output_small
                    del output_middle
                    del output_large
                    del output_below_large


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
                comb_soft_out = torch.cat([soft_out_seq_large,soft_out_seq_middle,soft_out_seq_small,soft_out_seq_below_large,soft_out_seq_extra_middle],dim=0)

                comb_soft_out_var = torch.var(comb_soft_out,dim=0,unbiased=True,keepdim=True)


                soft_out_ensem_mid = torch.where(summed_soft_out>=3,soft_out_ensem_ones,soft_out_ensem_zeros)

                gtv_var_map = comb_soft_out_var[0,1,:,:,:].squeeze().cpu().numpy()
                soft_label_map = soft_label_seq[0,1,:,:,:].squeeze().cpu().numpy()
                soft_out_ensem_mid_map = soft_out_ensem_mid[0,1,:,:,:].squeeze().cpu().numpy()
                input_slice_map = input_slice_seq[0,0,:,:,:].squeeze().cpu().numpy()
                data_dict = {'gtv_var_map':gtv_var_map,'label_map':soft_label_map,'pred_map':soft_out_ensem_mid_map,'input_map':input_slice_map}
                names_gtv_np = names[0].split('/')[-1].strip('.nii')
                # print(gtv_var_map.shape,soft_label_map.shape,soft_out_ensem_mid_map.shape,input_slice_map.shape)
                # np.savez(os.path.join(gtv_variance_np_dir,names_gtv_np),data_dict)                



                gtv_dice_small = get_classwise_dice(soft_out_seq_small,soft_label_seq).cpu().numpy()
                gtv_dice_middle = get_classwise_dice(soft_out_seq_middle,soft_label_seq).cpu().numpy()
                gtv_dice_large = get_classwise_dice(soft_out_seq_large,soft_label_seq).cpu().numpy()
                gtv_dice_extra_middle = get_classwise_dice(soft_out_seq_extra_middle,soft_label_seq).cpu().numpy()
                gtv_dice_below_large = get_classwise_dice(soft_out_seq_below_large,soft_label_seq).cpu().numpy()

                gtv_dice_ensem_mid = get_classwise_dice(soft_out_ensem_mid,soft_label_seq).cpu().numpy()
                
                gtv_dice_out_ave = get_classwise_dice(soft_out_ave_seq,soft_label_seq).cpu().numpy()

                avg_test_dsc_small.append(gtv_dice_small)
                avg_test_dsc_middle.append(gtv_dice_middle)
                avg_test_dsc_large.append(gtv_dice_large)
                avg_test_dsc_extra_middle.append(gtv_dice_extra_middle)
                avg_test_dsc_below_large.append(gtv_dice_below_large)

                avg_test_dsc_ensem_mid.append(gtv_dice_ensem_mid)
                avg_test_dsc_out_ave.append(gtv_dice_out_ave)

                print('\n\n***************')
                print("Result in iter--{} for:{}, with Pixel-spacing:{}".format(iter_test,names,pixel_spacing))
                print("Result in iter--{} for:{}, with Pixel-spacing:{}".format(iter_test,names,pixel_spacing),file=infer_txt)
                print('Soft-out-shape',soft_out_seq_small.shape,'Soft Label-seq',soft_label_seq.shape)
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


        print('*****************************************')
        print("Done Overall Testing")
        print('*****************************************\n\n\n')
        print('*****************************************',file=infer_txt)
        print("Done Overall Testing",file=infer_txt)
        print('*****************************************\n\n\n',file=infer_txt)



    def run(self,exp,test_dataset,checkpoints):
        agent.__create_dataset(test_dataset)
        agent.__create_network()
        if(self.stage == 'train'):
            if self.config['training']['train_rand_crop']:
                self.__train(exp)
            else:
                self.__train(exp)
        elif self.stage == 'test':
            if len(checkpoints) == 0:
                self.__infer(exp,None)
            else:
                for checkpoint in checkpoints:
                    if self.config['network']['net_type'] == 'UNet2D5':                    
                        checkpoint = f"unet2d5_{checkpoint}.pt"
                    elif self.config['network']['net_type'] == 'VNet':
                        checkpoint = f"vnet_{checkpoint}.pt"
                    elif self.config['network']['net_type'] == 'UNet3D':
                        checkpoint = f"unet3d_{checkpoint}.pt"
                    elif self.config['network']['net_type'] == 'UNet3DGenesis':
                        checkpoint = f"unet3dgenesis_{checkpoint}.pt"
                    else:
                        checkpoint = f"unet2d5_{checkpoint}.pt"
                    checkpoint = os.path.join(self.config['testing']['checkpoint_dir'],checkpoint)
                    if not os.path.isfile(checkpoint):
                        print('No checkpoint',checkpoint)
                        continue
                    self.__infer2(exp,checkpoint)
        elif self.stage=='ensemble':
            self.__infer_ensemble(exp)


import random
import time
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
    data_scale = str(sys.argv[4])
    data_fold = str(sys.argv[5])

    sleep_time = 20.0
    print("Suspending for:{} seconds, In order to Wait for Previous Processes to End".format(sleep_time))
    time.sleep(sleep_time)
    torch.cuda.empty_cache()
    print("Commencing Work")   


    if stage =='test':
        checkpoints = [i*50 for i in range(1,100)]
        try:        
            test_dataset = str(sys.argv[4])
        except:
            test_dataset = 'uncut'
    else:
        checkpoints = []
        test_dataset = 'none'
        

    config   = parse_config(cfg_file)
    agent    = TrainInferAgent(
                                config=config, 
                                data_fold=data_fold,
                                data_scale=data_scale,
                                stage=stage,
                                exp=exp,
                                )
    agent.run(exp,test_dataset,checkpoints)

# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
import torch
import pandas as pd
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pymic.io.image_read_write import load_image_as_nd_array

from pymic.self_supervised_tasks.preprocess.preprocess_rpl import PreprocessRPL
from pymic.self_supervised_tasks.preprocess.preprocess_rotation import PreprocessRotation
from pymic.self_supervised_tasks.preprocess.preprocess_matching import PreprocessMatching
from pymic.self_supervised_tasks.preprocess.preprocess_rpl_rot_exemp import *
from pymic.io.transform3d import *

class NiftyDataset_SSL(Dataset):
    """Dataset for loading images. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors
    with dimention order [C, H, W] for 2D images"""

    def __init__(self, root_dir, csv_file,transform='rpl'):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with image names.
            modal_num (int): Number of modalities.
            with_label (bool): Load the data with segmentation ground truth.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir   = root_dir
        self.csv_items  = pd.read_csv(csv_file)
        self.transform  = transform

        if transform == 'rpl':
            self.transform = PreprocessRPL()
        elif transform == 'rotation':
            self.transform = PreprocessRotation()
        else:
            raise('which-transform')        

    def __len__(self):
        return len(self.csv_items)

    def __getitem__(self, idx):
        names_list, image_list = [], []
        image_name = self.csv_items.iloc[idx,0]
        image_full_name = "{0:}/{1:}".format(self.root_dir, image_name)
        image_dict = load_image_as_nd_array(image_full_name)
        image_data = image_dict['data_array']
        image_data = image_data.squeeze().transpose(2, 1, 0)
        names_list.append(image_name)
        image_list.append(image_data)

        image = np.concatenate(image_list, axis = 0)
        image = np.asarray(image, np.float32)
        image = np.expand_dims(image,axis=0)
        image,labels = self.transform(image)
        # print(image.shape,'transform')
        sample = {'image': image.copy(), 'label':labels.copy(), 
                  'names' : names_list[0]}

        return sample


class NiftyDataset_SSL_Match(Dataset):
    """Dataset for loading images. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors
    with dimention order [C, H, W] for 2D images"""

    def __init__(self, root_dir, csv_file):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with image names.
            modal_num (int): Number of modalities.
            with_label (bool): Load the data with segmentation ground truth.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir   = root_dir
        self.csv_items  = pd.read_csv(csv_file)
        self.transform  = PreprocessMatching()


    def __len__(self):
        return len(self.csv_items)

    def __getitem__(self, idx):
        names_list, image_list = [], []
        image_name = self.csv_items.iloc[idx,0]
        image_full_name = "{0:}/{1:}".format(self.root_dir, image_name)
        image_dict = load_image_as_nd_array(image_full_name)
        image_data = image_dict['data_array']
        image_data = image_data.squeeze().transpose(2, 1, 0)
        names_list.append(image_name)
        image_list.append(image_data)

        image = np.concatenate(image_list, axis = 0)
        image = np.asarray(image, np.float32)
        image = np.expand_dims(image,axis=0)
        image = self.transform(image)
        # print(image.shape,'transform')
        sample = {'image': image.copy(),'names' : names_list[0]}

        return sample

class NiftyDataset_SSL_RPL_ROT_Exemp(Dataset):
    """Dataset for loading images. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors
    with dimention order [C, H, W] for 2D images"""

    def __init__(self, root_dir, csv_file):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with image names.
            modal_num (int): Number of modalities.
            with_label (bool): Load the data with segmentation ground truth.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir   = root_dir
        self.csv_items  = pd.read_csv(csv_file)
        self.transform  = PreprocessRPLRotExemp()
        self.safe_idx = []


    def __len__(self):
        return len(self.csv_items)

    def __getitem__(self, idx): 
        names_list, image_list = [], []
        image_name = self.csv_items.iloc[idx,0]
        image_full_name = "{0:}/{1:}".format(self.root_dir, image_name)
        image_dict = load_image_as_nd_array(image_full_name)
        image_data = image_dict['data_array']
        image_data = image_data.squeeze().transpose(2, 1, 0)
        names_list.append(image_name)
        image_list.append(image_data)

        image = np.concatenate(image_list, axis = 0)
        image = np.asarray(image, np.float32)
        image = np.expand_dims(image,axis=0)
        img_rot,lab_rot,img_rpl,lab_rpl,combined_crop_img = self.transform(image)
        sample = {'rpl':{'image':img_rpl.copy(),'label':lab_rpl.copy()}, 
                  'rot':{'image':img_rot.copy(),'label':lab_rot.copy()},
                  'exemplar':combined_crop_img.copy(),
                  }
        # sample = {  'rpl-image':img_rpl.copy(),
        #             'rpl-label':lab_rpl.copy(),
        #             'rot-image':img_rot.copy(),
        #             'rot-label':lab_rot.copy(),
        #             'exemplar':combined_crop_img.copy()            
        # }
        
        # print(img_rpl.shape,lab_rpl.shape)
        # print(img_rot.shape,lab_rot.shape)
        # print(combined_crop_img.shape)
        

        return sample
    
class NiftyDataset_SSL_RPL_ROT(Dataset):
    """Dataset for loading images. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors
    with dimention order [C, H, W] for 2D images"""

    def __init__(self, root_dir, csv_file):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with image names.
            modal_num (int): Number of modalities.
            with_label (bool): Load the data with segmentation ground truth.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir   = root_dir
        self.csv_items  = pd.read_csv(csv_file)
        self.transform  = PreprocessRPLRot()
        self.safe_idx = []


    def __len__(self):
        return len(self.csv_items)

    def __getitem__(self, idx): 
        names_list, image_list = [], []
        image_name = self.csv_items.iloc[idx,0]
        image_full_name = "{0:}/{1:}".format(self.root_dir, image_name)
        image_dict = load_image_as_nd_array(image_full_name)
        image_data = image_dict['data_array']
        image_data = image_data.squeeze().transpose(2, 1, 0)
        names_list.append(image_name)
        image_list.append(image_data)

        image = np.concatenate(image_list, axis = 0)
        image = np.asarray(image, np.float32)
        image = np.expand_dims(image,axis=0)
        img_rot,lab_rot,img_rpl,lab_rpl = self.transform(image)
        sample = {'rpl':{'image':img_rpl.copy(),'label':lab_rpl.copy()}, 
                  'rot':{'image':img_rot.copy(),'label':lab_rot.copy()},}
        

        return sample
    
NORMAL_TRAIN_DIR = "/media/disk1/jansen/code_rad/Dataset_Rad2/gtv_normal_processed_uni/small_scale"
class NiftyDataset_SSL_RPL_ROT2(Dataset):
    """Dataset for loading images. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors
    with dimention order [C, H, W] for 2D images"""

    def __init__(self, root_dir, data_paths):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with image names.
            modal_num (int): Number of modalities.
            with_label (bool): Load the data with segmentation ground truth.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir   = root_dir
        self.transform  = PreprocessRPLRot()
        self.safe_idx = []
        self.img_paths = data_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx): 
        names_list, image_list = [], []
        # image_name = self.csv_items.iloc[idx,0]
        # image_full_name = "{0:}/{1:}".format(self.root_dir, image_name)
        image_full_name = self.img_paths[idx]
        image_name = image_full_name.split('/')[-1]
        image_dict = load_image_as_nd_array(image_full_name)
        image_data = image_dict['data_array']
        image_data = image_data.squeeze().transpose(2, 1, 0)
        names_list.append(image_name)
        image_list.append(image_data)

        image = np.concatenate(image_list, axis = 0)
        image = np.asarray(image, np.float32)
        image = np.expand_dims(image,axis=0)
        img_rot,lab_rot,img_rpl,lab_rpl = self.transform(image)
        sample = {'rpl':{'image':img_rpl.copy(),'label':lab_rpl.copy()}, 
                  'rot':{'image':img_rot.copy(),'label':lab_rot.copy()},}
        

        return sample

class NiftyDataset_SSL2(Dataset):
    """Dataset for loading images. It generates 4D tensors with
    dimention order [C, D, H, W] for 3D images, and 3D tensors
    with dimention order [C, H, W] for 2D images"""

    def __init__(self, root_dir, data_paths,transform='rpl'):
        """
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with image names.
            modal_num (int): Number of modalities.
            with_label (bool): Load the data with segmentation ground truth.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir   = root_dir
        self.safe_idx = []
        self.img_paths = data_paths

        if transform == 'rpl':
            self.transform = PreprocessRPL()
        elif transform == 'rotation':
            self.transform = PreprocessRotation()
        else:
            raise('which-transform')        

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        names_list, image_list = [], []
        # image_name = self.csv_items.iloc[idx,0]
        # image_full_name = "{0:}/{1:}".format(self.root_dir, image_name)
        image_full_name = self.img_paths[idx]
        image_name = image_full_name.split('/')[-1]
        image_dict = load_image_as_nd_array(image_full_name)
        image_data = image_dict['data_array']
        image_data = image_data.squeeze().transpose(2, 1, 0)
        names_list.append(image_name)
        image_list.append(image_data)

        image = np.concatenate(image_list, axis = 0)
        image = np.asarray(image, np.float32)
        image = np.expand_dims(image,axis=0)
        image,labels = self.transform(image)
        # print(image.shape,'transform')
        sample = {'image': image.copy(), 'label':labels.copy(), 
                  'names' : names_list[0]}

        return sample
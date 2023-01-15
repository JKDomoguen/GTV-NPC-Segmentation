import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from nilearn.image import resample_img
import skimage.transform as skTrans

import nibabel as nib
import cv2
import csv
import imageio
from tqdm import tqdm
from ipywidgets import *
from PIL import Image

from preprocess import load_origin_nifty_volume_as_array

def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath)
    print(ct_scan.header.get_zooms())
    array   = ct_scan.get_fdata()
    # print(array.shape)
    # array   = np.array(array)
    array   = np.rot90(np.array(array))

    return(array)



def transform_data(img):
    None


def get_mask(dat_path):
    None

def get_img_data(data_path):
    None

if __name__ == "__main__":
    DATA_DIR = '../../../Dataset_Rad/liver-tumor/ct_scans'
    LABEL_DIR = '../../../Dataset_Rad/liver-tumor/ct_masks'
    # with open('liver_image_all.csv', 'w', encoding='UTF8', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['data','label'])
    for data_file in os.listdir(DATA_DIR):
        label_file = data_file.replace('volume','segmentation')
        label_arr = load_origin_nifty_volume_as_array(os.path.join(LABEL_DIR,label_file))
        print(set(label_arr[0].astype(int).flatten().tolist()) )
        # break

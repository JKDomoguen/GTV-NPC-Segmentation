import numpy as np
import random
from pymic.util.image_process import *
from pymic.Toolkit.deform import *
from pymic.self_supervised_tasks.preprocess.crop import crop_patches_3d
from pymic.self_supervised_tasks.preprocess.preprocess_rotation import rotate_image_3d

from pymic.self_supervised_tasks.preprocess.pad import *
from pymic.self_supervised_tasks.preprocess.crop import *

def preprocess_image_3d(image, patches_per_side, patch_jitter, is_training):
    cropped_image = crop_patches_3d(image, is_training, patches_per_side, patch_jitter)
    return np.array(cropped_image)


def preprocess_batch_3d(batch,  patches_per_side, patch_jitter=0, is_training=True):
    shape = batch.shape
    batch_size = shape[0]
    patch_count = patches_per_side ** 3

    labels = np.zeros((batch_size, patch_count - 1))
    patches = []

    center_id = int(patch_count / 2)

    for batch_index in range(batch_size):
        cropped_image = preprocess_image_3d(batch[batch_index], patches_per_side, patch_jitter, is_training)

        class_id = np.random.randint(patch_count - 1)
        patch_id = class_id
        if class_id >= center_id:
            patch_id = class_id + 1

        if is_training:
            image_patches = np.array([cropped_image[center_id], cropped_image[patch_id]])
            patches.append(image_patches)
        else:
            patches.append(np.array(cropped_image))

        labels[batch_index, class_id] = 1

    return np.array(patches), np.array(labels)

def preprocess_single_image(image,patches_per_side, patch_jitter=0, is_training=True):
    image = image.squeeze()
    patch_count = patches_per_side ** 3

    labels = np.zeros((patch_count - 1))

    center_id = int(patch_count / 2)

    cropped_image = preprocess_image_3d(image, patches_per_side, patch_jitter, is_training)

    class_id = np.random.randint(patch_count - 1)
    patch_id = class_id
    if class_id >= center_id:
        patch_id = class_id + 1

    if is_training:
        image_patches = np.array([cropped_image[center_id], cropped_image[patch_id]])

    else:
        image_patches = np.array(cropped_image)

    labels[class_id] = 1
    image_patches = np.expand_dims(image_patches,axis=1)
    return image_patches, np.array(labels)

class Pad(object):
    """
    Pad the image (shape [C, D, H, W] or [C, H, W]) to an new spatial shape, 
    the real output size will be max(image_size, output_size)
    Args:
       output_size (tuple/list): the size along each spatial axis. 
       
    """
    def __init__(self, output_size, inverse = True):
        self.output_size = output_size
        self.inverse = inverse


    def __call__(self, image):
        input_shape = image.shape
        input_dim = len(input_shape) - 1
        assert(len(self.output_size) == input_dim)
        margin = [max(0, self.output_size[i] - input_shape[1+i]) \
            for i in range(input_dim)]

        margin_lower = [int(margin[i] / 2) for i in range(input_dim)]
        margin_upper = [margin[i] - margin_lower[i] for i in range(input_dim)]
        pad = [(margin_lower[i], margin_upper[i]) for  i in range(input_dim)]
        pad = tuple([(0, 0)] + pad)
        if(max(margin) > 0):
            image = np.pad(image, pad, 'reflect')   

        
        return image

class RandomCrop(object):
    """Randomly crop the input image (shape [C, D, H, W] or [C, H, W]) 
    Args:
        output_size (tuple or list): Desired output size [D, H, W] or [H, W].
            the output channel is the same as the input channel.
    """

    def __init__(self, output_size, inverse = True):
        assert isinstance(output_size, (list, tuple))
        self.output_size = output_size
        self.inverse = inverse

    def __call__(self, image):
        input_shape = image.shape
        input_dim   = len(input_shape) - 1

        assert(input_dim == len(self.output_size))
        crop_margin = [input_shape[i + 1] - self.output_size[i]\
            for i in range(input_dim)]
        crop_min = [random.randint(0, item) for item in crop_margin]
        crop_max = [crop_min[i] + self.output_size[i] \
            for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max
        image = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
       
        return image

def augment_exemplar_3d(image):
    # prob to apply transforms
    alpha = 0.5
    beta = 0.5
    gamma = 0.15  # takes way too much time

    rotate_only_90 = 0.5

    def _distort_zoom(scan):
        scan_shape = scan.shape
        factor = 0.2
        zoom_factors = [np.random.uniform(1 - factor, 1 + factor) for _ in range(scan.ndim - 1)] + [1]
        scan = ndimage.zoom(scan, zoom_factors, mode="constant")
        scan = pad_to_final_size_3d(scan, scan_shape[0])
        scan = crop_3d(scan, True, scan_shape)
        return scan

    def _distort_color(scan):
        """
        This function is based on the distort_color function from the tf implementation.
        :param scan: image as np.array
        :return: processed image as np.array
        """
        # adjust brightness
        max_delta = 0.125
        delta = np.random.uniform(-max_delta, max_delta)
        scan += delta

        # adjust contrast
        lower = 0.5
        upper = 1.5
        contrast_factor = np.random.uniform(lower, upper)
        scan_mean = np.mean(scan)
        scan = (contrast_factor * (scan - scan_mean)) + scan_mean
        return scan

    processed_image = image.copy()
    for i in range(3):
        if np.random.rand() < 0.5:
            processed_image = np.flip(processed_image, i)

    # make rotation arbitrary instead of multiples of 90deg
    if np.random.rand() < alpha:
        if np.random.rand() < rotate_only_90:
            processed_image = np.rot90(processed_image, k=np.random.randint(0, 4), axes=(0, 1))
        else:
            processed_image = ndimage.rotate(processed_image, np.random.uniform(0, 360), axes=(0, 1), reshape=False)

    if np.random.rand() < alpha:
        if np.random.rand() < rotate_only_90:
            processed_image = np.rot90(processed_image, k=np.random.randint(0, 4), axes=(1, 2))
        else:
            processed_image = ndimage.rotate(processed_image, np.random.uniform(0, 360), axes=(1, 2), reshape=False)

    if np.random.rand() < alpha:
        if np.random.rand() < rotate_only_90:
            processed_image = np.rot90(processed_image, k=np.random.randint(0, 4), axes=(0, 2))
        else:
            processed_image = ndimage.rotate(processed_image, np.random.uniform(0, 360), axes=(0, 2), reshape=False)

    if np.random.rand() < beta:
        # color distortion
        processed_image = _distort_color(processed_image)
    if np.random.rand() < gamma:
        # zooming
        processed_image = _distort_zoom(processed_image)

    return processed_image

class PreprocessRPLRotExemp(object):
    '''
    Transform RPL object batch
    '''
    def __init__(self,
                 patches_per_side=3,
                 patch_jitter=3,
                 output_size_rot=(48,48,48),
                 output_size_rpl=(57,153,153),
                #  output_size_exemp=(48,64,64),
                #  output_size=(57,201,201)
                 ):
        self.patch_jitter = patch_jitter
        self.patches_per_side = patches_per_side
        self.patch_xyz = patches_per_side
        self.output_size = output_size_rpl
        self.padder = Pad(output_size_rpl)
        self.rand_cropper_rpl = RandomCrop(output_size_rpl)
        self.rand_cropper_rot = RandomCrop(output_size_rot)
    

    def __call__(self,image):
        input_shape = image.shape
        input_dim   = len(input_shape) - 1
        assert(input_dim == len(self.output_size))
        pad_image = self.padder(image)
        
        cropped_image_rpl = self.rand_cropper_rpl(pad_image)
        image_patches_rpl,labels_rpl = preprocess_single_image(cropped_image_rpl,self.patches_per_side,self.patch_jitter)

        cropped_image_rot = self.rand_cropper_rot(pad_image)
        image_patches_rot,labels_rot = rotate_image_3d(cropped_image_rot)
        image_patches_rot = np.expand_dims(image_patches_rot.squeeze(),axis=0)
        
        
        augment_image = augment_exemplar_3d(cropped_image_rot)
        
        augment_image = np.expand_dims(augment_image.squeeze(),axis=0)
        if augment_image.shape == cropped_image_rot.shape:
            combined_crop_img = np.stack([cropped_image_rot,augment_image]).astype(np.float32)
        else:
            combined_crop_img = np.stack([cropped_image_rot,image_patches_rot]).astype(np.float32)

        return image_patches_rot,labels_rot,image_patches_rpl,labels_rpl,combined_crop_img

class PreprocessRPLRot(object):
    '''
    Transform RPL object batch
    '''
    def __init__(self,
                 patches_per_side=3,
                 patch_jitter=3,
                 output_size_rot=(48,48,48),
                 output_size_rpl=(57,153,153),
                 ):
        self.patch_jitter = patch_jitter
        self.patches_per_side = patches_per_side
        self.patch_xyz = patches_per_side
        self.output_size = output_size_rpl
        self.padder = Pad(output_size_rpl)
        self.rand_cropper_rpl = RandomCrop(output_size_rpl)
        self.rand_cropper_rot = RandomCrop(output_size_rot)
    

    def __call__(self,image):
        input_shape = image.shape
        input_dim   = len(input_shape) - 1
        assert(input_dim == len(self.output_size))
        pad_image = self.padder(image)
        
        cropped_image_rpl = self.rand_cropper_rpl(pad_image)
        image_patches_rpl,labels_rpl = preprocess_single_image(cropped_image_rpl,self.patches_per_side,self.patch_jitter)

        cropped_image_rot = self.rand_cropper_rot(pad_image)
        image_patches_rot,labels_rot = rotate_image_3d(cropped_image_rot)
        image_patches_rot = np.expand_dims(image_patches_rot.squeeze(),axis=0)
        

        return image_patches_rot,labels_rot,image_patches_rpl,labels_rpl
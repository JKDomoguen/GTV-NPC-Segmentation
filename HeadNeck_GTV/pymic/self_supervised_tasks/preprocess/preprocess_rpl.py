import numpy as np
import random
from pymic.util.image_process import *
from pymic.Toolkit.deform import *
from pymic.self_supervised_tasks.preprocess.crop import crop_patches_3d




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

class PreprocessRPL(object):
    '''
    Transform RPL object batch
    '''
    def __init__(self,
                 patches_per_side=3,
                 patch_jitter=3,
                #  output_size=(57,105,105)
                 output_size=(57,153,153)
                #  output_size=(57,201,201)
                 ):
        self.patch_jitter = patch_jitter
        self.patches_per_side = patches_per_side
        self.patch_xyz = patches_per_side
        self.output_size = output_size
        self.padder = Pad(output_size)
        self.rand_cropper = RandomCrop(output_size)
    

    def __call__(self,image):
        input_shape = image.shape
        input_dim   = len(input_shape) - 1
        assert(input_dim == len(self.output_size))
        # print('\n\ninput-shape',input_shape)
        pad_image = self.padder(image)
        # print('pad-image',pad_image.shape)
        cropped_image = self.rand_cropper(pad_image)
        # print('cropped-image',cropped_image.shape)
        # cropped_image = image


        image_patches,labels = preprocess_single_image(cropped_image,self.patches_per_side,self.patch_jitter)
        # print("Final-image-patche",image_patches.shape)
        # print('\n\n')
        return image_patches,labels
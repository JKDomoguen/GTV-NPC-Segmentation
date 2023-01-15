import numpy as np
import random
from pymic.util.image_process import *
from pymic.Toolkit.deform import *
from pymic.self_supervised_tasks.preprocess.crop import crop_patches_3d



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


import random
class PreprocessMatching(object):
    '''
    Transform RPL object batch
    '''
    def __init__(self,
                 patches_per_side=3,
                 patch_jitter=3,
                #  output_size=(57,105,105)
                #  output_size=(57,153,153)
                #  output_size=(57,201,201)
                 output_size=(32,64,64)
                 ):
        self.patch_xyz = patches_per_side
        self.output_size = output_size
        self.padder = Pad(output_size)
        self.rand_cropper = RandomCrop(output_size)
        out_sz_np = np.array(output_size)
        cent_out_sz_np = (out_sz_np/2)/2
        half_out_sz_np = (out_sz_np/4)/2
        quarter_out_sz_np = out_sz_np/2

        min_half_out_sz = tuple(  np.absolute((cent_out_sz_np - half_out_sz_np).astype(int)) )
        max_half_out_sz = tuple((cent_out_sz_np + half_out_sz_np).astype(int) )
        min_quarter_out_sz = tuple( np.absolute((cent_out_sz_np - quarter_out_sz_np).astype(int)) )
        max_quarter_out_sz = tuple( (cent_out_sz_np + quarter_out_sz_np).astype(int) )
        # print(min_quarter_out_sz,max_quarter_out_sz)
        quarter_mask = np.zeros(output_size)
        quarter_mask[ np.ix_(range(min_quarter_out_sz[0], max_quarter_out_sz[0]),
                      range(min_quarter_out_sz[1], max_quarter_out_sz[1]),
                      range(min_quarter_out_sz[2], max_quarter_out_sz[2])) ] = 1
        half_mask = np.zeros(output_size)

        half_mask[  
                     np.ix_(range(min_half_out_sz[0], max_half_out_sz[0]),
                     range(min_half_out_sz[1], max_half_out_sz[1]),
                     range(min_half_out_sz[2], max_half_out_sz[2]))
                     ] = 1        
        
        self.quarter_mask = np.expand_dims(quarter_mask.astype(int),axis=0)
        self.half_mask = np.expand_dims(half_mask.astype(int),axis=0)
        self.not_quarter_mask = np.expand_dims( (np.invert(quarter_mask.astype(bool))).astype(int),axis=0)
        self.not_half_mask = np.expand_dims( (np.invert(half_mask.astype(bool))).astype(int),axis=0)

    def __call__(self,image):
        input_shape = image.shape
        input_dim   = len(input_shape) - 1
        assert(input_dim == len(self.output_size))
        pad_image = self.padder(image)
        cropped_image = self.rand_cropper(pad_image)
        if random.randint(0,1):
            inner_cropped = self.half_mask*cropped_image
            outer_cropped = self.not_half_mask*cropped_image
        else:
            inner_cropped = self.quarter_mask*cropped_image
            outer_cropped = self.quarter_mask*cropped_image

        # print(inner_cropped.shape,outer_cropped.shape,cropped_image.shape)

        combined_crop_img = np.stack([cropped_image,inner_cropped,outer_cropped]).astype(np.float32)

        return combined_crop_img
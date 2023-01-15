import numpy as np
import albumentations as ab
from pymic.util.image_process import *
from pymic.Toolkit.deform import *
import random

def rotate_batch_3d(x, y=None):
    batch_size = x.shape[0]
    y = np.zeros((batch_size, 10))
    rotated_batch = []
    for index, volume in enumerate(x):
        rot = np.random.random_integers(10) - 1

        if rot == 0:
            volume = volume
        elif rot == 1:
            volume = np.transpose(np.flip(volume, 1), (1, 0, 2, 3))  # 90 deg Z
        elif rot == 2:
            volume = np.flip(volume, (0, 1))  # 180 degrees on z axis
        elif rot == 3:
            volume = np.flip(np.transpose(volume, (1, 0, 2, 3)), 1)  # 90 deg Z
        elif rot == 4:
            volume = np.transpose(np.flip(volume, 1), (0, 2, 1, 3))  # 90 deg X
        elif rot == 5:
            volume = np.flip(volume, (1, 2))  # 180 degrees on x axis
        elif rot == 6:
            volume = np.flip(np.transpose(volume, (0, 2, 1, 3)), 1)  # 90 deg X
        elif rot == 7:
            volume = np.transpose(np.flip(volume, 0), (2, 1, 0, 3))  # 90 deg Y
        elif rot == 8:
            volume = np.flip(volume, (0, 2))  # 180 degrees on y axis
        elif rot == 9:
            volume = np.flip(np.transpose(volume, (2, 1, 0, 3)), 0)  # 90 deg Y

        rotated_batch.append(volume)
        y[index, rot] = 1
    return np.stack(rotated_batch), y

def rotate_image_3d(image, y=None):
    y = np.zeros(10)
    rot = np.random.random_integers(10) - 1

    volume = image.transpose(2,3,1,0)
    # print('before rotation',volume.shape)


    if rot == 0:
        volume = volume
    elif rot == 1:
        volume = np.transpose(np.flip(volume, 1), (1, 0, 2, 3))  # 90 deg Z
    elif rot == 2:
        volume = np.flip(volume, (0, 1))  # 180 degrees on z axis
    elif rot == 3:
        volume = np.flip(np.transpose(volume, (1, 0, 2, 3)), 1)  # 90 deg Z
    elif rot == 4:
        volume = np.transpose(np.flip(volume, 1), (0, 2, 1, 3))  # 90 deg X
    elif rot == 5:
        volume = np.flip(volume, (1, 2))  # 180 degrees on x axis
    elif rot == 6:
        volume = np.flip(np.transpose(volume, (0, 2, 1, 3)), 1)  # 90 deg X
    elif rot == 7:
        volume = np.transpose(np.flip(volume, 0), (2, 1, 0, 3))  # 90 deg Y
    elif rot == 8:
        volume = np.flip(volume, (0, 2))  # 180 degrees on y axis
    elif rot == 9:
        volume = np.flip(np.transpose(volume, (2, 1, 0, 3)), 0)  # 90 deg Y

    y[rot] = 1

    # print(volume.shape,rot,'after-rotation')
    return volume, y

def resize(batch, new_size):
    return np.array([ab.Resize(new_size, new_size)(image=image)["image"] for image in batch])

class PreprocessRotation(object):
    def __init__(self,output_size=(32,32,32)):
        assert isinstance(output_size, (list, tuple))
        self.output_size = output_size
    def __call__(self,image):
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
        cropped_image = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
        image_patches,labels = rotate_image_3d(cropped_image)
        image_patches = np.expand_dims(image_patches.squeeze(),axis=0)
        return image_patches,labels
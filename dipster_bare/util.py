import torch
import torch.nn as nn
# import torchvision
import sys

import numpy as np
# from PIL import Image
# import PIL
import numpy as np

from math import pi
import mrcfile

def np_to_torch(img_np, dev):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    #if is numpy array
    if isinstance(img_np, np.ndarray):
        return torch.from_numpy(img_np).to(dev)
    else:
        return img_np

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    #if is torch tensor
    if isinstance(img_var, torch.Tensor):
        img_var = img_var.detach().cpu().numpy()
    return img_var

def grs(angle_min = -70, angle_max = 70, N = 70):

    gr = (1 + np.sqrt(5)) / 2
    range_rad = np.radians(angle_max - angle_min)
    indices = np.arange(1, N + 1)   # index starts at 1 by default

    angles = np.degrees(
        np.mod(indices * gr * range_rad, range_rad) + np.radians(angle_min)
    ).round(2)

    return angles

def read_mrcfile(fname):
    with mrcfile.open(fname, permissive = True) as mrc:
        return mrc.data
    
def normalize_min_max(vol):
    vol = vol.astype(np.float32)
    return (vol - vol.min()) / (vol.max() - vol.min())

def normalize_precentile(vol, p1 = 2, p2 = 98):
    vol = vol.astype(np.float32)
    p1, p2 = np.percentile(vol, [p1, p2])
    return (vol - p1) / (p2 - p1)
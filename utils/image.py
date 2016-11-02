# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:00:10 2016

@author: HongXiang

Process images:
    convert to gray image
    crop image into patches    
Storage save into .npy files
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from scipy import misc
import random

def imread(filename):
    return misc.imread(filename)

def rgb2gray(image):    
    """
    Calculate gray image from RGB image.
    Args:
        image: a [N*]H*W[*C] tensor, if N is provided, then C must provide.
    """
    if len(image.shape) == 4:
        grayimg = np.mean(image, axis=3, keepdims=True)
    if len(image.shape) == 3:
        grayimg = np.mean(image, axis=2)
    if len(image.shape) == 2:
        grayimg = np.copy(image)
    return grayimg

def down_sample(img, down_sample_factor = 2):
    height = img.shape[0]
    width = img.shape[1]
    height_down = height // down_sample_factor
    width_down = width // down_sample_factor
    if len(img.shape) == 2:
        shape_down = [height_down, width_down]
    else:
        shape_down = [height_down, width_down, img.shape[3]]
    img_down = misc.imresize(img, shape_down)
    img_down = misc.imresize(img_down, img.shape)
    return img_down

def patch_generator(image, patch_height, patch_width, stride_v, stride_h):   
    """
    WARNNING: DeprecationWarning
    """
    image_height = image.shape[0]
    image_width = image.shape[1] 
    if len(image.shape) == 3:        
        with_multi_channel = True
    else:        
        with_multi_channel = False
    x_offset = 0
    y_offset = 0
    while y_offset + patch_height < image_height:
        while x_offset + patch_width < image_width:
            if with_multi_channel:
                crop = image[y_offset:y_offset+patch_height, x_offset:x_offset+patch_width, :]
            else:
                crop = image[y_offset:y_offset+patch_height, x_offset:x_offset+patch_width]
            yield crop
            x_offset = x_offset + stride_h
            if x_offset > image_width:
                x_offset = image_width - patch_width
        x_offset = 0                                       
        y_offset = y_offset + stride_v
        if y_offset > image_height:
            y_offset = image_height - patch_height

def patch_random_generator(image, patch_height, patch_width, n_patches):
    """
    WARNNING: DeprecationWarning
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    assert(image_height >= patch_height)
    assert(image_width >= patch_width)
    if len(image.shape) == 3:
        with_multi_channel = True
    else:        
        with_multi_channel = False
    n_cropped = 0    
    
    while n_cropped < n_patches:
        y_offset = random.randint(0, image_height-patch_height)
        x_offset = random.randint(0, image_width-patch_width)        
        if with_multi_channel:
            crop = image[y_offset:y_offset+patch_height, x_offset:x_offset+patch_width, :]
        else:
            crop = image[y_offset:y_offset+patch_height, x_offset:x_offset+patch_width]
        yield crop
        n_cropped += 1


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
    grayimg = np.mean(image,2)
    if len(grayimg.shape) == 2:
        grayimg = np.reshape(grayimg, grayimg.shape + [1])
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

def offset_generator(image_shape, patch_shape, stride_step):
    image_height = image_shape[0]
    image_width = image_shape[1]
    patch_height = patch_shape[0]
    patch_width = patch_shape[1]
    stride_v = stride_step[0]
    stride_h = stride_step[1]                    
    x_offset_list = []
    y_offset_list = []
    x_offset = 0
    y_offset = 0
    while y_offset + patch_height <= image_height:
        while x_offset + patch_width <= image_width:
            x_offset_list.append(x_offset)
            y_offset_list.append(y_offset)
            x_offset += stride_h
        if x_offset < image_width:
            x_offset_list.append(image_width-patch_width)
            y_offset_list.append(y_offset)
        x_offset = 0
        y_offset += stride_v
    if y_offset < image_height:
        x_offset = 0
        y_offset = image_height-patch_height
        while x_offset + patch_width <= image_width:
            x_offset_list.append(x_offset)
            y_offset_list.append(y_offset)
            x_offset += stride_h
        if x_offset < image_width:
            x_offset_list.append(image_width-patch_width)
            y_offset_list.append(y_offset)
    return x_offset_list, y_offset_list

def patch_generator_tensor(tensor, patch_shape, stride_step, n_patches=None, use_random_shuffle=False):
    """Full functional patch generator
    Args:
    tensor: a N*W*H*C shape tensor
    """    
    assert(len(tensor.shape)==4)    
    image_shape = [tensor.shape[1], tensor.shape[2]]
    x_offset_list, y_offset_list = offset_generator(image_shape, patch_shape, stride_step)    
    ids = list(xrange(len(x_offset_list)))
    if use_random_shuffle:
        random.shuffle(ids)
    if n_patches == None:
        n_patches = len(ids)
    ids = ids[:n_patches]
    for i in ids:
        x_offset = x_offset_list[i]
        y_offset = y_offset_list[i]
        patch = tensor[:, y_offset: y_offset+patch_shape[0], x_offset: x_offset+patch_shape[1], :]
        yield patch

def patches_recon_tensor(patch_list, tensor_shape, patch_shape, stride_step, valid_shape, valid_offset):
    """
    Iterpolator: later
    """
    image_shape = [tensor_shape[1], tensor_shape[2]]
    x_offset, y_offset=offset_generator(image_shape, patch_shape, stride_step)
    tensor=np.zeros(tensor_shape)
    cid = 0
    for patch in patch_list:
        ty0 = y_offset[cid] + valid_offset[0] + 12
        tx0 = x_offset[cid] + valid_offset[1] + 12
        py0 = valid_offset[0]
        px0 = valid_offset[1]
        dy = valid_shape[0]
        dx = valid_shape[1]
        tensor[:, ty0: ty0+dy, tx0: tx0+dx, :] = patch[:, py0: py0+dy, px0: px0+dx, :]
        tensor[:, ty0: ty0+25, tx0: tx0+25, :] = patch[:, py0: py0+25, px0: px0+25, :]        
        cid += 1
    return tensor
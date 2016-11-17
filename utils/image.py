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
import random
import numpy as np
from scipy import misc
from six.moves import xrange
import xlearn.utils.general as utg

def imread(filename):
    return misc.imread(filename)

def image_type(tensor):
    if len(tensor.shape) == 2:
        return 'gray'
    if len(tensor.shape) == 3 and tensor.shape[2] == 3:
        return 'RGB'
    if len(tensor.shape) == 3 and tensor.shape[0] == 1:
        return '1gray'
    if len(tensor.shape) == 3 and tensor.shape[2] == 1:
        return 'gray1'
    if len(tensor.shape) == 3:
        return 'Ngray'
    if len(tensor.shape) == 4 and tensor.shape[3] == 1 and tensor.shape[0] == 1:
        return '1gray1'
    if len(tensor.shape) == 4 and tensor.shape[3] == 1:
        return 'Ngray1'
    if len(tensor.shape) == 4 and tensor.shape[3] == 3:
        return 'NRGB'
    if len(tensor.shape) == 4:
        return 'NHWC'
    if len(tensor.shape) == 5:
        return 'NHWDC'
    return 'unknown'


def rgb2gray(image):
    """
    Calculate gray image from RGB image.
    Args:
        image: a [N*]H*W[*C] tensor, if N is provided, then C must provide.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(utg.errmsg(type(image), np.ndarray, "Wrong input type, "))
    if len(image.shape) < 2 or len(image.shape) > 5:
        raise TypeError(utg.errmsg(len(image.shape), (2, 3, 4), "Wrong input dimension, "))
    if len(image.shape) == 4:
        grayimg = np.mean(image, axis=3, keepdims=True)
    if len(image.shape) == 3:
        grayimg = np.mean(image, axis=2)
    if len(image.shape) == 2:
        grayimg = np.copy(image)
    return grayimg

def proj2sino(input_):
    height = input_.shape[0]
    width = input_.shape[1]
    nangles = input_.shape[2]
    output = []
    for iz in xrange(height):
        sinogram = np.zeros([width, nangles])
        for iwidth in range(width):
            for iangle in range(nangles):
                sinogram[iwidth, iangle] = input_[iz, iwidth, iangle]
        output.append(sinogram)

def sino2proj(input_):
    if not hasattr(input_, "__iter__"):
        input_ = [input_]
    height = len(input_)
    width, nangles = input_[0].shape
    output = np.zeros([height, width, nangles])
    for iz in xrange(height):
        for iwidth in range(width):
            for iangle in range(nangles):
                output[iz, iwidth, iangle] = input_[iz][iwidth, iangle]
    return output

def image_formater(input_, is_uint8=False, offset=3):
    if not isinstance(input_, np.ndarray):
            raise TypeError('ImageFormater only accepts {0}, given {1}'.format(
                np.ndarray, type(input_)))
    img_type = xlearn.utils.tensor.image_type(input_)
    output = input_
    if img_type is 'gray' or img_type is 'RGB':
        output = input_
    if img_type is 'gray1':
        output = input_[:, :, 0]
    if img_type is '1gray1':
        output = input_[0, :, :, 0]
    if img_type is 'Ngray':
        if input_.shape[0] == 1:
            output = input_[0, :, :]
        else:
            output = xlearn.utils.tensor.multi_image2large_image(
                input_, offset=self._offset)
    if img_type is 'NRGB':
        if input_.shape[0] == 1:
            output = input_[0, :, :, :]
        else:
            output = xlearn.utils.tensor.multi_image2large_image(
                input_, offset=self._offset)
    if self._is_uint8:
        output = np.uint8(output)
    return output

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


"""Useful routines for handling tensor
"""
from __future__ import absolute_import, division, print_function
from six.moves import xrange

import matplotlib.pyplot as plt
import numpy as np
import scipy
import h5py

def merge_patch_list(patch_list):
    """Merge a list of tensors into a large tensor
    Args: tensor list
    Return: large tensor
    """
    patch_shape = patch_list[0].shape
    if len(patch_shape) == 4:
        n_single_patch = patch_shape[0]
    else:
        n_single_patch = 1
    n_patch = len(patch_list)
    tensor_shape = []
    tensor_shape.append(n_patch*n_single_patch)
    if len(patch_shape) == 4:
        tensor_shape += patch_shape[1:]
    else:
        tensor_shape += patch_shape
    tensor = np.zeros(tensor_shape)
    with_channel = len(tensor_shape) == 4
    for id_patch in xrange(n_patch):        
        if with_channel:
            tensor[id_patch*n_single_patch:(id_patch+1)*n_single_patch, :, :, :] = patch_list[id_patch]
        else:
            tensor[id_patch*n_single_patch:(id_patch+1)*n_single_patch, :, :] = patch_list[id_patch]
    return tensor

def multi_image2large_image(multi_img_tensor, id_list=None, offset=1, n_image_row=None):
    """Change a tensor into a large image
    Args:
        multi_img_tensor: a tensor in N*H*W*[3,4] form.
    Return:
        a large image
    """
    shape = multi_img_tensor.shape
    if id_list is None:
        id_list = list(xrange(shape[0]))
    n_img = len(id_list)
    height = shape[1]
    width = shape[2]
    
    if len(shape) == 3:
        n_channel = 1
    else:
        n_channel = shape[3]
    tensor_formated = np.zeros([n_img, height, width, n_channel])
    for i in id_list:
        if len(shape) == 3:
            tensor_formated[i, :, :, 0] = multi_img_tensor[id_list[i], :, :]
        else:
            tensor_formated[i, :, :, :] = multi_img_tensor[id_list[i], :, :, :]
    if n_image_row==None:
        n_image_row = int(np.ceil(np.sqrt(n_img)))
    n_image_col = int(np.ceil(n_img/n_image_row))

    img_large = np.zeros([n_image_col*(height+offset)+offset, n_image_row*(width+offset)+offset, n_channel])
    for i_channel in range(n_channel):
        for i_patch in range(n_img):    
            [row, col] = np.unravel_index(i_patch, [n_image_col, n_image_row])
            x_offset = col*(width+offset)+offset
            y_offset = row*(height+offset)+offset
            img_large[y_offset:y_offset+height, x_offset:x_offset+width, i_channel] = tensor_formated[i_patch, :, :, i_channel]
    return img_large

def split_channel(tensor_multi_channel, id_N_list = None, id_C_list = None):
    """Reshape a tensor of dim N and dim channel.
    Args:
        multi_img_tensor: a tensor in 1*H*W*C form.
    Return:
        a large image
    """
    shape = tensor_multi_channel.shape
    assert len(shape) == 4
    n_img = shape[0]
    n_channel = shape[3]
    if id_N_list is None:
        id_N_list = list(xrange(n_img))
    if id_C_list is None:
        id_C_list = list(xrange(n_channel))

    height = shape[1]
    width = shape[2]
    multi_channel_form = np.zeros([n_img, height, width, n_channel])
    for i in xrange(n_img):
        for j in xrange(n_channel):
            multi_channel_form[i, :, :, j] = tensor_multi_channel[id_N_list[i], :, :, id_C_list[j]]
    n_img_axis = int(np.ceil(np.sqrt(n_channel)))
    img_large = np.zeros([n_img, n_img_axis*height+n_img_axis+1, n_img_axis*width+n_img_axis+1])
    for i_img in range(n_img):
        for i_channel in range(n_channel):
            [iy, ix] = np.unravel_index(i_channel, [n_img_axis, n_img_axis])
            x_offset = ix*(width+1)+1
            y_offset = iy*(height+1)+1
            img_large[i_img, y_offset:y_offset+height, x_offset:x_offset+width] = multi_channel_form[i_img, :, :, i_channel]
    return img_large
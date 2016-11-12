"""Useful routines for handling tensor
"""
from __future__ import absolute_import, division, print_function
from six.moves import xrange

import matplotlib.pyplot as plt
import numpy as np
import scipy
import h5py
import random

def image_type(tensor):    
    if len(tensor.shape) == 2:
        return 'gray'
    if len(tensor.shape) == 3 and tensor.shape[2] == 3:
        return 'RGB'        
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
    

def unpack_tensor_ndlist(tensor_list):
    result = []
    for tensor_maybe in tensor_list:
        if isinstance(tensor_maybe, np.ndarray):
            result.append(tensor_maybe)
        else:
            if not isinstance(tensor_maybe, list):
                raise TypeError('Must be tensor, or [recurrent] list of tensor, not '+type(tensor_maybe) + '.')
            result.extend(unpack_tensor_ndlist(tensor_maybe))
    return result

def merge_tensor_list(tensor_list, squeeze=False):
    """Merge a list of tensors into a large tensor
    supports tensor up to 5 dimension
    Args: tensor list
    Return: a tensor with +1 dimension at first dimension
    """    
    tensor_shape = tensor_list[0].shape
    if len(tensor_shape) > 5:
        raise TypeError("Merging tensor with dim %d, maximum 5."%len(tensor_shape))
    n_tensor = len(tensor_list)
    if n_tensor > 1:    
        for i in xrange(n_tensor):    
            if tensor_list[i].shape != tensor_shape:
                raise TypeError("Tensor with shape {0} merge one with shape {1}.".format(tensor_shape, tensor_list[i].shape))
    newshape = [n_tensor]+list(tensor_shape)    
    tensor_o = np.zeros(newshape)    
    for i in xrange(n_tensor):        
        if len(tensor_shape) == 5:
            tensor_o[i, :, :, :, :, :] = tensor_list[i]
        if len(tensor_shape) == 4:
            tensor_o[i, :, :, :, :] = tensor_list[i]
        if len(tensor_shape) == 3:
            tensor_o[i, :, :, :] = tensor_list[i]
        if len(tensor_shape) == 2:
            tensor_o[i, :, :] = tensor_list[i]
        if len(tensor_shape) == 1:
            tensor_o[i, :] = tensor_list[i]
    output_shape = list(tensor_o.shape)
    while 1 in output_shape:
        output_shape.remove(1)
    tensor_o = np.reshape(tensor_o, output_shape)    
    return tensor_o

def merge_patch_list(patch_list):
    """Merge a list of tensors into a large tensor
    Args: tensor list
    Return: large tensor
    """
    """
    WARNING: DeprecationWarning
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

def patches_recon_tensor(patch_list,
                         tensor_shape, patch_shape, stride_step,
                         valid_shape, valid_offset):
    """
    Iterpolator: later
    """
    if len(tensor_shape) != 4:
        raise TypeError('tensor_shape needs to be 4D, get %dD.'%len(tensor_shape))
    if len(patch_shape) != 2:
        raise TypeError('tensor_shape needs to be 2D, get %dD.'%len(patch_shape))
    if len(stride_step) != 2:
        raise TypeError('tensor_shape needs to be 2D, get %dD.'%len(stride_step))
    if len(valid_shape) != 2:
        raise TypeError('tensor_shape needs to be 2D, get %dD.'%len(valid_shape))
    if len(valid_offset) != 2:
        raise TypeError('tensor_shape needs to be 2D, get %dD.'%len(valid_offset))

    image_shape = [tensor_shape[1], tensor_shape[2]]    
    x_offset, y_offset=offset_generator(image_shape, patch_shape, stride_step)    
    tensor=np.zeros(tensor_shape)
    cid = 0
    for patch in patch_list:
        # ty0 = y_offset[cid] + valid_offset[0] + 12        
        # tx0 = x_offset[cid] + valid_offset[1] + 12
        ty0 = y_offset[cid] + valid_offset[0]
        tx0 = x_offset[cid] + valid_offset[1]
        py0 = valid_offset[0]
        px0 = valid_offset[1]
        dy = valid_shape[0]
        dx = valid_shape[1]
        tensor[:, ty0: ty0+dy, tx0: tx0+dx, :] = patch[:, py0: py0+dy, px0: px0+dx, :]
        # tensor[:, ty0: ty0+25, tx0: tx0+25, :] = patch[:, py0: py0+25, px0: px0+25, :]        
        cid += 1
    return tensor
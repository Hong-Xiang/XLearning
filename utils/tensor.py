"""Useful routines for handling tensor
"""
from __future__ import absolute_import, division, print_function
from six.moves import xrange

import numpy as np
import scipy
import h5py
import random
import copy


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
    raise DeprecationWarning()
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
    raise DeprecationWarning()
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
    raise DeprecationWarning()
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
    raise DeprecationWarning()
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

def offset_1d(length, patch_size, stride=1, offset0=0, offset1=0, check_all=False):
    """Slicing offset generator for 1-D array.
    Args:
        length :: Int
    Return:
        None
    """
    offset = offset0
    while offset + patch_size - 1 < length - offset1:
        yield offset
        offset += stride
    if check_all:
        offset -= stride
        if offset + patch_size - 1 < length - offset1 - 1:
            offset = length - offset1 - patch_size
            yield offset

def offset_nd(tensor_shape, patch_shape, strides=None, offsets0=None, offsets1=None, check_all=False):
    """Slicing offset generator for n-D array.
    Args:
        length :: Int
    Return:
        None
    """
    dim = len(tensor_shape)
    if dim == 0:
        yield []
    else:
        if strides is None:
            strides = [1]*dim
        if offsets0 is None:
            offsets0 = [0]*dim
        if offsets1 is None:
            offsets1 = [0]*dim
        crop = lambda x: (x[0], x[1:])
        tensor_shape0, tensor_shape_next = crop(tensor_shape)
        patch_shape0, patch_shape_next = crop(patch_shape)
        strides0, strides_next = crop(strides)
        offset00, offsets0_next = crop(offsets0)
        offset10, offsets1_next = crop(offsets1)
        for offset in offset_1d(length=tensor_shape0,
                                patch_size=patch_shape0,
                                stride=strides0,
                                offset0=offset00,
                                offset1=offset10,
                                check_all=check_all):
            for offset_list in offset_nd(tensor_shape_next,
                                         patch_shape_next,
                                         strides_next,
                                         offsets0_next,
                                         offsets1_next,
                                         check_all=check_all):
                yield [offset]+offset_list

def shape_dim_fix(shape_input, shape_to_embbed, dim0=0, dim1=0):
    dim_input = len(shape_input)
    dim_embbed = len(shape_to_embbed)
    dimstart = dim_embbed - dim1 - dim_input
    if dimstart < dim0:
        check_pass = False
    while dimstart >= dim0:
        check_pass = True
        for j in xrange(dim_input):
            if shape_input[j] > shape_to_embbed[dimstart+j]:
                check_pass = False
                break
        if check_pass:
            break
        dimstart -= 1
    if not check_pass:
        raise ValueError("Wrong dimension. Input:{0}, Embed:{1}, dim0:{2}, dim1:{3}.".format(
            shape_input, shape_to_embbed, dim0, dim1))
    output = [1] * dim_embbed
    output[dimstart:dimstart+dim_input] = shape_input
    return output

def multidim_slicer(index_start, index_range, strides=None):
    """
    Caution: true end is index_start + index_range
    """
    output = []
    dim = len(index_start)
    if strides is None:
        strides = [1]*dim
    if dim != len(index_range):
        raise ValueError("Wrong dimension.")
    for i in xrange(dim):
        output.append(slice(index_start[i], index_start[i]+index_range[i], strides[i]))
    output = tuple(output)
    return output


def crop_tensor(tensor, patch_shape, strides=None, margin0=None, margin1=None, check_all=False):
    output = []
    for offset in offset_nd(tensor.shape,
                            patch_shape,
                            strides=strides,
                            offsets0=margin0,
                            offsets1=margin1,
                            check_all=check_all):
        sli = multidim_slicer(offset, patch_shape)
        output.append(tensor[sli])
    return output

def combine_tensor_list(tensor_list, shape, strides=None, margin0=None, margin1=None, dim0=0, dim1=0, check_all=False):
    """combine tensor list into a larger tensor
    Try to put those tensors into tensor, in C order.
    """
    dim = len(shape)
    output = np.zeros(shape)
    if len(tensor_list) == 0:
        return output
    if not isinstance(tensor_list, list):
        raise TypeError("Input must be a {0}, got {1}.".format(list, type(tensor_list)))
    if not isinstance(tensor_list[0], np.ndarray):
        raise TypeError("Element must be a {0}, got {1}.".format(np.ndarray, type(tensor_list[0])))
    element_shape = tensor_list[0].shape
    fixed_shape = shape_dim_fix(element_shape, shape, dim0, dim1)
    if strides is None:
        strides = [0]*dim
    if margin0 is None:
        margin0 = [0]*dim
    if margin1 is None:
        margin1 = [0]*dim
    strides = copy.copy(fixed_shape)
    for i in xrange(dim):
        strides[i] += strides[i]
    cid = 0
    for strides in offset_nd(shape, fixed_shape, strides, margin0, margin1, check_all):
        sli = multidim_slicer(strides, fixed_shape)
        output[sli] = np.reshape(tensor_list[cid], fixed_shape)
        cid += 1
        if cid == len(tensor_list):
            break
    return output

def down_sample_1d(input_, axis, ratio, offset=0, method='mean'):
    """Down sample a tensor on given axis with ratio.
    """
    input_shape = input_.shape
    dim = len(input_shape)
    output_shape = list(input_shape)
    output_shape[axis] //= ratio
    if method == 'fixed':
        index_start = [0]*dim
        index_range = list(input_shape)
        strides = [1]*dim
        index_start[axis] = offset+ratio//2
        index_range[axis] = output_shape[axis]*ratio
        strides[axis] = ratio
        sli = multidim_slicer(index_start, index_range, strides)
        output = np.zeros(output_shape)
        output[:] = input_[sli]
    if method == 'mean':
        index_start = [0]*dim
        index_range = list(input_shape)
        strides = [1]*dim
        strides[axis] = ratio
        output = np.zeros(output_shape)
        for step in xrange(ratio):
            index_start[axis] = offset + step
            index_range[axis] = output_shape[axis]*ratio
            sli = multidim_slicer(index_start, index_range, strides)
            output = output + input_[sli]
        output /= ratio
    return output

def down_sample_nd(input_, ratio, offset=None, method='mean'):
    dim = len(input_.shape)
    if offset is None:
        offset = [0]*dim
    # Check dimensions
    # TODO: implement dimension checks.

    output = np.zeros(input_.shape)
    output[:] = input_

    for axis in xrange(dim):
        output = down_sample_1d(output, axis=axis, ratio=ratio[axis], offset=offset[axis], method=method)
    return output


def offset_generator(image_shape, patch_shape, stride_step):
    """
    """
    raise DeprecationWarning()
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

def patch_generator_tensor(tensor, patch_shape, stride_step, n_patches=None, use_random_shuffle=False, threshold=None):
    raise DeprecationWarning()
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
    outputed = 0
    for i in ids:
        x_offset = x_offset_list[i]
        y_offset = y_offset_list[i]
        patch = tensor[:, y_offset: y_offset+patch_shape[0], x_offset: x_offset+patch_shape[1], :]
        if threshold is not None:
            nnz = np.float(np.count_nonzero(patch))/patch.size
            if nnz < threshold:
                continue
        yield patch
        outputed += 1
        if outputed > n_patches:
            break

def patches_recon_tensor(patch_list,
                         tensor_shape, patch_shape, stride_step,
                         valid_shape, valid_offset):
    """
    
    Iterpolator: later
    """
    raise DeprecationWarning()
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

"""Useful routines for handling tensor
"""
from __future__ import absolute_import, division, print_function
from six.moves import xrange

import numpy as np
import scipy
import h5py
import random
import copy

import xlearn.utils.general as utg


def to_bin(x, threshold=None, zero_value=0, one_value=1):
    if threshold is None:
        threshold = (np.min(x) + np.max(x)) / 2    
    y = np.zeros(shape=x.shape, dtype=np.int32)
    y[x < threshold] = zero_value
    y[x >= threshold] = one_value
    return y


def to_cata(x, thresholds=None, nb_threshold=None, values=None):
    if thresholds is None:
        if nb_threshold is None:
            raise ValueError(
                "Both thresholds and nb_threshold are None, only one of them can be None.")
        thresholds = np.linspace(np.min(x), np.max(x), nb_threshold + 1)
    nb_threshold = len(thresholds) - 1
    y = np.zeros(shape=x.shape, dtype=np.int32)
    if values is None:
        values = list(range(nb_threshold))
    y[x == thresholds[-1]] = values[-1]
    for i in range(nb_threshold):
        y[np.logical_and(x < thresholds[i + 1], x >= thresholds[i])] = values[i]
    return y


def load_mat(filename, varnames):
    file_in = h5py.File(filename, 'r')
    output = []
    for name in varnames:
        output.append(np.array(file_in.get(name)))


def offset_1d(length, patch_size, stride=1, offset0=0, offset1=0, check_all=False):
    """Slicing offset generator for 1-D array.
    Args:
        length :: int, length of array to be split,
        patch_size :: int, patch size,
        stride :: int,
        offset0 :: int, left offset
        offset1 :: int, right offset
        check_all :: True for force include end point
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
            strides = [1] * dim
        if offsets0 is None:
            offsets0 = [0] * dim
        if offsets1 is None:
            offsets1 = [0] * dim
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
            if dim > 1:
                for offset_list in offset_nd(tensor_shape_next,
                                             patch_shape_next,
                                             strides_next,
                                             offsets0_next,
                                             offsets1_next,
                                             check_all=check_all):
                    if not hasattr(offset_list, "__iter__"):
                        output = [offset, offset_list]
                    else:
                        output = [offset] + offset_list
                    yield output
            else:
                yield offset


def shape_dim_fix(shape_input, shape_to_embbed, dim0=0, dim1=0, n_items=1, smart=False):
    """Refine shape issues to embbed multiple small tensor into a larger one.
    There are two works done by this function:
    1.  extend dimension of shape_input to the dimension of shape_to_embbed by
        adding one to axis if necessary
    2.  refine size of shape_to_embbed if given shape has 'None', '-1' or list elements.
    Args:
        shape_to_embbed: shape of tensor to embbed in, a list of integers or
        list of integers.
            - >0: largest size of given axis, allow multiple items.
            - 0: to refine. set to shape_input correspoinding size.
            - -1: to refine. set to fit n_items, allow multiple items.
              Only *ONE* -1 is allowed.
            - list of integers: to refine. special allowed size.
    Return:
        (refined input shape, refined embed shape)
    """
    shape_to_embbed = list(shape_to_embbed)
    n_mo = shape_to_embbed.count(-1)
    if n_mo > 1:
        raise ValueError(
            "# of -1 must less than one, given {0}.".format(shape_to_embbed))
    # shape_input_refine = []
    # for dimc in shape_input:
    #     if dimc is None:
    #         shape_input_refine.append(None)
    #         continue
    #     if not hasattr(dimc, "__iter__"):
    #         shape_input_refine.append(dimc)
    # shape_input = shape_input_refine
    if smart:
        if len(shape_input) == 2:
            shape_to_embbed = [-1, 0, 0, [1]]
        elif len(shape_input) == 3:
            shape_to_embbed = [-1, 0, 0, [1, 3]]
        elif len(shape_input) == 4 and shape_input[3] in [1, 3]:
            shape_to_embbed = [-1, 0, 0, [1, 3]]
        elif len(shape_input) == 4:
            shape_to_embbed = [-1, 0, 0, 0]
        elif len(shape_input) == 5:
            shape_to_embbed = [-1, 0, 0, 0, [1, 3]]
    dim_input = len(shape_input)
    dim_embbed = len(shape_to_embbed)
    dimstart = dim_embbed - dim1 - dim_input
    if dimstart < dim0:
        check_pass = False
    free_dim = 1
    while dimstart >= dim0:
        size_mo = np.float(n_items)
        check_pass = True
        for j in xrange(dim_input):
            if shape_to_embbed[dimstart + j] == 0:
                continue
            if hasattr(shape_to_embbed[dimstart + j], "__iter__"):
                if not shape_input[j] in shape_to_embbed[dimstart + j]:
                    check_pass = False
                    break
                continue
            elif shape_to_embbed[dimstart + j] == -1:
                free_dim = shape_input[j]
                continue
            elif shape_input[j] > shape_to_embbed[dimstart + j]:
                check_pass = False
                break
            item_this_dim = shape_to_embbed[
                dimstart + j] // shape_input[j]
            if item_this_dim == 0:
                item_this_dim = 1
            size_mo = np.ceil(size_mo / item_this_dim)
        if check_pass:
            break
        dimstart -= 1
    if not check_pass:
        raise ValueError("Wrong dimension. Input:{0}, Embed:{1}, dim0:{2}, dim1:{3}.".format(
            shape_input, shape_to_embbed, dim0, dim1))
    output_input = [1] * dim_embbed
    output_input[dimstart: dimstart + dim_input] = shape_input
    output_embbed = shape_to_embbed[:]
    for j in xrange(dim_input):
        if shape_to_embbed[dimstart + j] == 0:
            output_embbed[dimstart + j] = shape_input[j]
            continue
        if hasattr(shape_to_embbed[dimstart + j], "__iter__"):
            output_embbed[dimstart + j] = shape_input[j]
            continue
        if shape_to_embbed[dimstart + j] == -1:
            width = size_mo * free_dim
            output_embbed[dimstart + j] = int(width)
    output_embbed_refine = []
    for dimsize in output_embbed:
        if hasattr(dimsize, "__iter__"):
            output_embbed_refine.append(dimsize[0])
            continue
        if dimsize == 0:
            output_embbed_refine.append(1)
            continue
        if dimsize == -1:
            width = size_mo * free_dim
            output_embbed_refine.append(int(width))
            continue
        output_embbed_refine.append(dimsize)
    output_embbed = output_embbed_refine
    return output_input, output_embbed


def multidim_slicer(index_start, index_range, strides=None):
    """
    Caution: true end is index_start + index_range
    A[index_start[0]:index_start[0]+index_range[0], index_start[1]:index_start[1]+index_range[1], ...]
    """
    if not hasattr(index_start, "__iter__"):
        index_start = [index_start]
    if not hasattr(index_range, "__iter__"):
        index_range = [index_range]
    output = []
    dim = len(index_start)
    if strides is None:
        strides = [1] * dim
    if dim != len(index_range):
        raise ValueError("Wrong dimension.")
    for i in xrange(dim):
        output.append(slice(index_start[i],
                            index_start[i] + index_range[i],
                            strides[i]))
    output = tuple(output)
    return output


def crop_tensor(tensor, patch_shape, strides=None, margin0=None, margin1=None,
                check_all=False, n_patches=None, random_shuffle=False):
    """Crop tensor into a list of small tensors
    """
    output = []
    offsets = []
    for offset in offset_nd(tensor.shape,
                            patch_shape,
                            strides=strides,
                            offsets0=margin0,
                            offsets1=margin1,
                            check_all=check_all):
        offsets.append(offset)
    if n_patches is None:
        n_patches = len(offsets)
    offsets = offsets[:n_patches]
    if random_shuffle:
        random.shuffle(offsets)
    for offset in offsets:
        sli = multidim_slicer(offset, patch_shape)
        output.append(tensor[sli])
    return output


def combine_tensor_list(tensor_list, shape, strides=None, margin0=None, margin1=None, dim0=0, dim1=0, check_all=False):
    """combine tensor list into a larger tensor
    Try to put those tensors into tensor, in C order.
    """
    if not isinstance(tensor_list, list):
        raise TypeError("Input must be a {0}, got {1}.".format(
            list, type(tensor_list)))
    if not isinstance(tensor_list[0], np.ndarray):
        raise TypeError("Element must be a {0}, got {1}.".format(
            np.ndarray, type(tensor_list[0])))
    n_tensor = len(tensor_list)
    if n_tensor == 0:
        raise ValueError("Empty imput list.")
    element_shape = tensor_list[0].shape
    patch_shape, embed_shape = shape_dim_fix(
        element_shape, shape, dim0, dim1, n_items=n_tensor)
    dim = len(embed_shape)
    output = np.zeros(embed_shape)
    if strides is None:
        strides = patch_shape
    else:
        strides = list(strides)
    if margin0 is None:
        margin0 = [0] * dim
    if margin1 is None:
        margin1 = [0] * dim
    cid = 0
    for i in xrange(dim):
        if margin0[i] < 0:
            raise ValueError(utg.errmsg(
                margin0, ">0", "margin0 needs to be positive"))
        if margin1[i] < 0:
            raise ValueError(utg.errmsg(
                margin0, ">0", "margin1 needs to be positive"))
    for offset in offset_nd(embed_shape, patch_shape, strides, margin0, margin1, check_all):

        sli = multidim_slicer(offset, patch_shape)

        output[sli] = np.reshape(tensor_list[cid], patch_shape)
        cid += 1
        if cid == len(tensor_list):
            break
    return output


def downsample_shape(input_shape, ratio):
    output_shape = [sz // r for (sz, r) in utg.zip_equal(input_shape, ratio)]
    return output_shape


def upsample_shape(input_shape, ratio):
    output_shape = [sz * r for (sz, r) in utg.zip_equal(input_shape, ratio)]
    return output_shape


def down_sample_1d(input_, axis, ratio, offset=0, method='mean'):
    """Down sample a tensor on given axis with ratio.
    """
    input_shape = input_.shape
    dim = len(input_shape)
    output_shape = list(input_shape)
    output_shape[axis] //= ratio
    if method == 'fixed':
        index_start = [0] * dim
        index_range = list(input_shape)
        strides = [1] * dim
        index_start[axis] = offset + (ratio - 1) // 2
        index_range[axis] = output_shape[axis] * ratio
        strides[axis] = ratio
        sli = multidim_slicer(index_start, index_range, strides)
        output = np.zeros(output_shape)
        output[:] = input_[sli]
    if method == 'mean' or method=='sum':
        index_start = [0] * dim
        index_range = list(input_shape)
        strides = [1] * dim
        strides[axis] = ratio
        output = np.zeros(output_shape)
        for step in xrange(ratio):
            index_start[axis] = offset + step
            index_range[axis] = output_shape[axis] * ratio
            sli = multidim_slicer(index_start, index_range, strides)
            output = output + input_[sli]
        if method == 'mean':
            output /= ratio
    return output


def down_sample_nd(input_, ratio, offset=None, method='mean'):
    """Down sample of tensor on N axises.
    """
    dim = len(input_.shape)
    if offset is None:
        offset = [0] * dim
    if dim != len(ratio):
        raise ValueError(utg.errmsg(len(ratio), dim),
                         "ratio dimension mismatch, ")
    if dim != len(offset):
        raise ValueError(utg.errmsg(len(offset), dim),
                         "offset dimension mismatch, ")
    output = np.zeros(input_.shape)
    output[:] = input_
    for axis in xrange(dim):
        output = down_sample_1d(output, axis=axis,
                                ratio=ratio[axis],
                                offset=offset[axis],
                                method=method)
    return output

# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 15:00:10 2016

@author: HongXiang

Process images:
    convert to gray image
    crop image into patches    
Storage save into .npy files
"""
import random
import numpy as np
from scipy import misc

import matplotlib.pyplot as plt

import xlearn.utils.general as utg
import xlearn.utils.tensor as utt

def shape_check(target, label=None):
    """ check whether target.shape is NHWC, and (optional) same as label.shape
    """
    target_shape = list(target.shape)
    if not len(target_shape) == 4:
        raise ValueError('Invalid target shape {}.'.format(target_shape))
    if label is not None:
        label_shape = list(label.shape)
    if not target_shape == label_shape:
        raise ValueError('Target shape {} differ to label shape {}.'.format(target_shape, label_shape))
    return None

def psnr(target, label, max_value=None, tor=1e-7):
    shape_check(target, label)
    if max_value is None:
        max_value = np.max([target, label])
    err_square = np.square(target - label)    
    err_sum = np.sum(err_square, axis=(1, 2, 3))        
    err_sum[err_sum < tor] = tor
    psnr_value = 10.0 * (1.0 + np.log(np.prod(target.shape[1:])*max_value*max_value/err_sum))    
    return psnr_value

def imshow_save_clean(img, filename, **kwargs):
    fig = plt.figure()
    fig.set_size_inches(img.shape[0]/100,img.shape[1]/100)
    fig.set_dpi(100)
    plt.imshow(img, **kwargs)
    plt.axis('off')
    plt.subplots_adjust(0,0,1,1,0,0)
    # plt.savefig(filename, bbox_inches='tight', pad_inches = 0)
    plt.savefig(filename)

def subplot_images(images, nb_max_row=8, cmap=None, is_gray=True, size=2.0, is_axis=False, tight_c=0.3, is_save=False, filename='images.png', window=None):
    """ subplot list of images of multiple categories into grid subplots
    Args:
        images: tuple of list of images
    """
    if isinstance(images, (tuple, list)):
        nb_cata = len(images)
    else:
        nb_cata = 1

    nb_images = len(images[0])
    nb_row = nb_images // nb_max_row
    nb_row = max(nb_row, 1)
    cid = 1
    fig = plt.figure(figsize=(nb_max_row * size, nb_row * nb_cata * size))
    if window is None:
        window = [(None, None)] * nb_cata
    for i in range(nb_row):
        for k in range(nb_cata):
            for j in range(nb_max_row):
                id_img = i * nb_max_row + j
                id_img = min(id_img, nb_images - 1)
                ax = plt.subplot(nb_row * nb_cata, nb_max_row, cid)                
                plt.imshow(images[k][id_img], cmap=cmap,
                           vmin=window[k][0], vmax=window[k][1])
                if is_gray:
                    plt.gray()
                if not is_axis:
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)
                cid += 1
    plt.tight_layout(tight_c)
    if is_save:
        fig.savefig(filename)


def imread(filename):
    return np.array(misc.imread(filename))


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
        image: a <N>*H*W*<C> tensor, if N is provided, then C must provide.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(utg.errmsg(
            type(image), np.ndarray, "Wrong input type, "))

    if len(image.shape) != 4:
        raise TypeError(utg.errmsg(len(image.shape),
                                   4, "Wrong input dimension, "))

    grayimg = np.mean(image, axis=3, keepdims=True)
    return grayimg


def proj2sino(input_):
    """SPECT projection to list of sinograms.
    """
    height = input_.shape[0]
    width = input_.shape[1]
    nangles = input_.shape[2]
    output = []
    for iheight in range(height):
        sinogram = np.zeros([width, nangles])
        for iwidth in range(width):
            for iangle in range(nangles):
                sinogram[iwidth, iangle] = input_[iheight, iwidth, iangle]
        output.append(sinogram)
    return output


def sino2proj(input_):
    """List of sinograms to SPECT projection
    """
    if not hasattr(input_, "__iter__"):
        input_ = [input_]
    height = len(input_)
    width, nangles = input_[0].shape
    output = np.zeros([height, width, nangles])
    for iheight in range(height):
        for iwidth in range(width):
            for iangle in range(nangles):
                output[iheight, iwidth, iangle] = input_[
                    iheight][iwidth, iangle]

    return output


def image2tensor(input_):
    """<image> or <list[image]> to <tensor> of shape NHWC
    """
    if isinstance(input_, np.ndarray):
        input_ = [input_]
    refined = []
    for image in input_:
        # image = np.squeeze(image)
        img_t = image_type(image)
        if img_t[0] == 'N':
            image_shape = list(image.shape)
            image_shape[0] = 1
            refined.extend(utt.crop_tensor(image, image_shape))
        else:
            refined.append(image)
    image_shape = refined[0].shape
    tensor_shape = list(image_shape)
    if len(image_shape) == 2:
        tensor_shape += [1]
    if len(image_shape) == 4 and tensor_shape[0] == 1:
        tensor_shape[0] = len(refined)
    else:
        tensor_shape = [len(refined)] + tensor_shape

    strides = [1, image_shape[0], image_shape[1], 1]
    output = utt.combine_tensor_list(refined, tensor_shape, strides=strides)
    return output


def tensor2image(input_, id_list=None, offset=3, n_image_row=None):
    """Change a tensor into a large image
    Args:
        input_: a tensor in N*H*W*<1 or 3> form.
    Return:
        a large image
    """
    shape = input_.shape
    if id_list is None:
        id_list = list(range(shape[0]))
    if len(id_list) == 1:
        id_list = [[id_list[0]]]
    n_img = len(id_list)
    height = shape[1]
    width = shape[2]
    channel = shape[3]
    input_ = input_[id_list, slice(0, height), slice(
        0, width), slice(0, channel)]

    image_shape = [1, height, width, channel]
    image_list = utt.crop_tensor(input_, image_shape)
    if n_image_row is None:
        n_image_row = int(np.ceil(np.sqrt(n_img)))
    n_image_col = int(np.ceil(n_img / n_image_row))

    new_height = n_image_col * (height + offset) + offset
    new_width = n_image_row * (width + offset) + offset
    new_shape = [1, new_height, new_width, channel]
    strides = [1, height + offset, width + offset, 1]

    margin0 = [0, offset, offset, 0]
    margin1 = [0, offset, offset, 0]
    output = utt.combine_tensor_list(
        image_list, new_shape, strides=strides, margin0=margin0, margin1=margin1)

    return output


def split_channel(input_, id_N_list=None, id_C_list=None, offset=3, n_image_row=None):
    raise DeprecationWarning()
    """Reshape a tensor of dim N and dim channel.
    Args:
        multi_img_tensor: a tensor in 1*H*W*C form.
    Return:
        a large image
    """
    if id_N_list is None:
        id_N_list = list(range(input_.shape[0]))
    if id_C_list is None:
        id_C_list = list(range(input_.shape[3]))
    if len(id_N_list) == 1:
        id_N_list = [[id_N_list[0]]]
    if len(id_C_list) == 1:
        id_C_list = [[id_C_list[0]]]
    height = input_.shape[1]
    width = input_.shape[2]
    input_ = input_[id_N_list, slice(0, height), slice(0, width), id_C_list]
    channel = input_.shape[3]
    patch_shape = [1, height, width, 1]
    strides = [1, height, width, 1]
    patch_list = utt.crop_tensor(input_, patch_shape, strides)
    new_shape = [len(patch_list), height, width, 1]
    changed = utt.combine_tensor_list(new_shape, new_shape, strides)
    output = tensor2image(changed, changed, offset, n_image_row)
    return output


def image_formater(input_, is_uint8=None, offset=3):
    """Format input into a plotable image.
    Args:
        input_:
            _ image
            - list of image
            - tensor NHWC
    """
    if not isinstance(input_, np.ndarray):
        raise TypeError(utg.errmsg(type(input_), np.ndarray),
                        "Wrong input type, ")

    tensor = image2tensor(input_)
    image = tensor2image(tensor, offset=offset)
    image = np.squeeze(image)
    if is_uint8 is None:
        if np.max(image) <= 256 and np.max(image) > 1:
            is_uint8 = True
        else:
            is_uint8 = False
    if is_uint8 is True:
        image = np.uint8(image)
    return image

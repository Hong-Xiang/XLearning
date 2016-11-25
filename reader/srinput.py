#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2016-11-03 10:20:52

@author: HongXiang

General image inputer
"""
from __future__ import absolute_import, division, print_function
import os
import itertools
import json
import numpy as np
from six.moves import xrange
import xlearn.utils.xpipes as utp
import xlearn.utils.tensor as utt
import xlearn.utils.general as utg


def label_name(data_name, case_digit=None, label_prefix=None):
    prefix, id, suffix = utg.seperate_file_name(data_name)
    if case_digit is not None:
        id = str(id)
        id = id[:-case_digit]
        id = int(id)
    if label_prefix is None:
        label_prefix = 'label'
    output = utg.form_file_name(label_prefix, id, suffix)
    return output


def config_file_generator(conf_file,
                          path_data, prefix_data,
                          path_label, prefix_label,
                          patch_shape,
                          strides,
                          batch_size,
                          same_file_data_label=False,
                          single_file=False,
                          filename=None,
                          n_patch_per_file=None,
                          down_sample_ratio=1,
                          ids=None,
                          padding=False,
                          dataset_type='test',
                          down_sample_method='fixed',
                          random_shuffle=False,
                          mean=1,
                          std=128,
                          **kwargs):
    """generate config .JSON file."""
    data = dict()
    data.update({'path_data': path_data})
    data.update({'prefix_data': prefix_data})
    data.update({'path_label': path_label})
    data.update({'prefix_label': prefix_label})
    data.update({'patch_shape': patch_shape})
    data.update({'strides': strides})
    data.update({'batch_size': batch_size})
    data.update({'same_file_data_label': same_file_data_label})
    data.update({'single_file': single_file})
    data.update({'filename': filename})
    data.update({'n_patches': n_patch_per_file})
    data.update({'down_sample_ratio': down_sample_ratio})
    data.update({'ids': ids})
    data.update({'padding': padding})
    data.update({'dataset_type': dataset_type})
    data.update({'down_sample_method': down_sample_method})
    data.update({'random_shuffle': random_shuffle})
    data.update({'need_gray': True})
    data.update({'check_all': False})
    data.update({'mean': mean})
    data.update({'std': std})
    data.update(kwargs)
    with open(conf_file, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4, separators=(',', ': '))


class DataSet(object):
    """
    A general super resolution net.
    Args:
    - config_file: a .JSON file containing configurations.
            if this

    next_batch() provides
            (low_resolution_patch_tensor, high_resolution_patch_tensor)
            #  path_data, prefix_data,
            #  path_label, prefix_label,
            #  patch_shape,
            #  strides,
            #  batch_size,
            #  same_file_data_label=False,
            #  single_file=False,
            #  filename,
            #  n_patch_per_file=None,
            #  down_sample_ratio=1,
            #  ids=None,
            #  padding=False,
            #  dataset_type='test',
            #  down_sample_method='fixed',
            #  mean=1,
            #  std=128):
    """

    def __init__(self, **kwargs):
        if "conf" in kwargs:
            with open(kwargs["conf"]) as conf_file:
                paras = json.load(conf_file)
        else:
            paras = kwargs

        self._patch_shape = paras['patch_shape']
        self._batch_size = paras['batch_size']
        self._dataset_type = paras['dataset_type']
        self._ratio = paras['down_sample_ratio']
        self._padding = paras['padding']
        self._dataset_type = paras['dataset_type']
        self._strides = paras['strides']
        self._std = paras['std']
        self._mean = paras['mean']
        self._data_filename_iter = utp.FileNameLooper(paras['path_data'],
                                                      prefix=paras[
                                                          'prefix_data'],
                                                      random_shuffle=paras[
                                                          'random_shuffle'],
                                                      ids=paras['ids'])
        if paras['same_file_data_label']:
            self._data_image = utp.NPYReaderSingle(self._data_filename_iter)
            self._data_image_copyer = utp.Copyer(
                self._data_image, copy_number=2)
            self._data_tensor = utp.TensorFormater(self._data_image_copyer)
            self._label_tensor = utp.TensorFormater(self._data_image_copyer)
        else:
            self._data_filename_copyer = utp.Copyer(
                self._data_filename_iter, copy_number=2)
            if paras['same_file_data_label']:
                self._label_filename = utp.Pipe(self._data_filename_copyer)
            else:
                self._label_filename = utp.LabelFinder(
                    self._data_filename_copyer, label_name)
            self._data_filename = utp.Pipe(self._data_filename_copyer)
            self._data_image = utp.NPYReaderSingle(self._data_filename)
            self._label_image = utp.NPYReaderSingle(self._label_filename)
            self._data_tensor = utp.TensorFormater(self._data_image)
            self._label_tensor = utp.TensorFormater(self._label_image)
        if paras['need_gray']:
            self._data_gray = utp.ImageGrayer(self._data_tensor)
            self._label_gray = utp.ImageGrayer(self._label_tensor)
            self._merge = utp.Pipe([self._data_gray, self._label_gray])
            self._stacker = utp.TensorStacker(self._merge)
        else:
            self._merge = utp.Pipe([self._data_tensor, self._label_tensor])
            self._stacker = utp.TensorStacker(self._merge)
        self._tensor = utp.TensorFormater(self._stacker)
        patch_shape = paras['patch_shape']
        patch_shape_2 = patch_shape[:]
        patch_shape_2[0] = 2

        self._patch_generator = utp.PatchGenerator(self._tensor,
                                                   shape=patch_shape_2,
                                                   n_patches=paras[
                                                       'n_patches'],
                                                   strides=paras[
                                                       'strides'],
                                                   random_gen=paras[
                                                       'random_shuffle'],
                                                   check_all=paras['check_all'])

        self._buffer = utp.Buffer(self._patch_generator)

        self._slicer = utp.TensorSlicer(self._buffer, patch_shape)

        self._buffer2 = utp.Buffer(self._slicer)

        self._hr_patch_gen = utp.TensorFormater(self._buffer2)

        self._down_sample = utp.DownSampler(
            self._buffer2, self._ratio, method=paras['down_sample_method'])

        self._lr_patch_gen = utp.TensorFormater(self._down_sample)

    def next_batch(self):
        high_shape = [self._batch_size, self._patch_shape[
            1], self._patch_shape[2], self._patch_shape[3]]
        if self._padding:
            low_height = int(np.ceil(self._patch_shape[1] / self._ratio[1]))
            low_width = int(np.ceil(self._patch_shape[2] / self._ratio[2]))
        else:
            low_height = int(self._patch_shape[1] / self._ratio[1])
            low_width = int(self._patch_shape[2] / self._ratio[2])
        low_shape = [self._batch_size, low_height,
                     low_width, self._patch_shape[3]]
        high_tensor = np.zeros(high_shape)
        low_tensor = np.zeros(low_shape)
        cid = 0        
        for i in xrange(self._batch_size):
            # for patch_high, patch_low in zip(self._hr_patch_gen.out,
            # self._lr_patch_gen.out):
            try:
                patch_high = next(self._hr_patch_gen.out)
                patch_low = next(self._lr_patch_gen.out)
            except StopIteration:
                break
            high_tensor[cid, :, :, :] = patch_high
            low_tensor[cid, :, :, :] = patch_low
            cid += 1

        low_tensor /= self._std
        low_tensor -= self._mean
        high_tensor /= self._std
        high_tensor -= self._mean

        return low_tensor, high_tensor

    def form_image(self, patches, image_shape, strides=None):
        if strides is None:
            strides = self._strides
        output = utt.combine_tensor_list(patches, image_shape, strides)
        return output

    @property
    def height(self):
        return self._patch_shape[0]

    @property
    def width(self):
        return self._patch_shape[1]

    @property
    def epoch(self):
        """current epoch"""
        return self._data_filename_iter.epoch

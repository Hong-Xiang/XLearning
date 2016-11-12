#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2016-11-03 10:20:52

@author: HongXiang

General image inputer
"""
from __future__ import absolute_import, division, print_function
import os
import numpy as np
from six.moves import xrange
import xlearn.utils.xpipes as xpipe
import xlearn.utils.dataset as xdata
import xlearn.utils.tensor



class DataSetSinogram(object):
    """
    A general super resolution net.

    next_batch() provides 
            (low_resolution_patch_tensor, high_resolution_patch_tensor)
    """

    def __init__(self, path, prefix,
                 patch_shape, strides,
                 batch_size,
                 n_patch_per_file=None,
                 down_sample_ratio=1,
                 ids=None,
                 padding=False,
                 dataset_type='test',
                 down_sample_method='fixed',
                 mean=1,
                 std=1e12):
        self._path = os.path.abspath(path)
        self._patch_shape = patch_shape
        self._batch_size = batch_size
        self._dataset_type = dataset_type
        self._n_patch = n_patch_per_file
        self._ratio = down_sample_ratio
        self._padding = padding
        if self._dataset_type == 'test':
            self._npyreader = xpipe.NPYReader(path, prefix,
                                              random_shuffle=False,
                                              ids=ids)
        if self._dataset_type == 'train':
            self._npyreader = xpipe.NPYReader(path, prefix,
                                              random_shuffle=True,
                                              ids=ids)

        self._tensor_rgb = xpipe.TensorFormater(self._npyreader,
                                                auto_shape=True)
        self._gray = xpipe.ImageGrayer(self._tensor_rgb)
        self._tensor_image = xpipe.TensorFormater(self._gray, auto_shape=True)
        if self._dataset_type == 'test':
            self._batch_generator = xpipe.PatchGenerator(self._tensor_image,
                                                         shape=self._patch_shape,
                                                         n_patches=self._n_patch,
                                                         strides=strides,
                                                         threshold=0.1)
        if self._dataset_type == 'train':
            self._batch_generator = xpipe.PatchGenerator(self._tensor_image,
                                                         shape=self._patch_shape,
                                                         random_gen=True,
                                                         n_patches=self._n_patch,
                                                         strides=strides,
                                                         threshold=0.1)
        self._buffer = xpipe.Buffer(self._batch_generator)
        self._copyer = xpipe.Copyer(self._buffer, copy_number=2)
        self._hr_patch_gen = xpipe.TensorFormater(
            self._copyer, auto_shape=True)
        self._down_sample = xpipe.DownSamplerSingle(
            self._copyer, axis=2, ratio=self._ratio, method=down_sample_method)
        self._lr_patch_gen = xpipe.TensorFormater(
            self._down_sample, auto_shape=True)
        self._std = std
        self._mean = mean

    def next_batch(self):

        high_shape = [self._batch_size, self._patch_shape[
            0], self._patch_shape[1], 1]

        if self._padding:
            low_height = self._patch_shape[0]
            low_width = int(np.ceil(self._patch_shape[1] / self._ratio))
        else:
            low_height = self._patch_shape[0]
            low_width = int(self._patch_shape[1] / self._ratio)

        low_shape = [self._batch_size, low_height, low_width, 1]
        high_tensor = np.zeros(high_shape)
        low_tensor = np.zeros(low_shape)
        added = 0
        while added < self._batch_size:        
            patch_high = self._hr_patch_gen.out.next()
            patch_low = self._lr_patch_gen.out.next()
            if np.max(patch_high) < 1:
                continue
            pstd = np.std(patch_high)
            patch_high /= pstd
            patch_low /= pstd
            pmean = np.mean(patch_high)
            patch_high -= pmean
            patch_low -= pmean
            high_tensor[added, :, :, :] = patch_high
            low_tensor[added, :, :, :] = patch_low
            added += 1

        return low_tensor, high_tensor

    @property
    def n_files(self):
        #TODO: Safely delete this method. (Or implement it.)
        pass

    @property
    def height(self):
        return self._patch_shape[0]

    @property
    def width(self):
        return self._patch_shape[1]

    @property
    def epoch(self):
        """current epoch"""
        return self._npyreader.epoch

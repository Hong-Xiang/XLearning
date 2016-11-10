#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2016-11-03 10:20:52

@author: HongXiang

General image inputer
"""
import os
import numpy as np
from six.moves import xrange
import xlearn.utils.xpipes as xpipe
import xlearn.utils.dataset as xdata
import xlearn.utils.tensor

class DataSet(object):
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
                 std=128):
        self._path = os.path.abspath(path)
        self._patch_shape = patch_shape
        self._batch_size = batch_size
        self._dataset_type = dataset_type
        self._n_patch = n_patch_per_file
        self._ratio = down_sample_ratio
        self._padding = padding        
        if self._dataset_type == 'test':
            self._npyreader = xpipe.NPYReader(path, prefix,                                                   
                                              random_shuffle=False)
        if self._dataset_type == 'train':
            self._npyreader = xpipe.NPYReader(path, prefix,                                                   
                                              random_shuffle=True)
        
        self._tensor_rgb = xpipe.TensorFormater(self._npyreader,
                                                  auto_shape=True)
        self._gray = xpipe.ImageGrayer(self._tensor_rgb)
        self._tensor_image = xpipe.TensorFormater(self._gray, auto_shape=True)
        if self._dataset_type == 'test':
            self._batch_generator = xpipe.PatchGenerator(self._tensor_image,
                                                         shape=self._patch_shape,
                                                         n_patches=self._n_patch,
                                                         strides=strides)
        if self._dataset_type == 'train':
            self._batch_generator = xpipe.PatchGenerator(self._tensor_image,
                                                         shape=self._patch_shape,
                                                         random_gen=True,
                                                         n_patches=self._n_patch,
                                                         strides=strides)        
        self._buffer = xpipe.Buffer(self._batch_generator)
        self._copyer = xpipe.Copyer(self._buffer, copy_number=2)
        self._hr_patch_gen = xpipe.TensorFormater(self._copyer, auto_shape=True)        
        self._down_sample = xpipe.DownSampler(self._copyer, self._ratio, method=down_sample_method)
        self._lr_patch_gen = xpipe.TensorFormater(self._down_sample, auto_shape=True)
        self._std = std
        self._mean = mean
        
    def next_batch(self):

        high_shape = [self._batch_size, self._patch_shape[0], self._patch_shape[1], 1]
        
        if self._padding:
            low_height = int(np.ceil(self._patch_shape[0]/self._ratio))
            low_width = int(np.ceil(self._patch_shape[1]/self._ratio))
        else:            
            low_height = int(self._patch_shape[0]/self._ratio)
            low_width = int(self._patch_shape[1]/self._ratio)
        
        low_shape = [self._batch_size, low_height, low_width, 1]                                                  
        high_tensor = np.zeros(high_shape)
        low_tensor = np.zeros(low_shape) 
        for i in xrange(self._batch_size):
            patch_high = self._hr_patch_gen.out.next()
            patch_low = self._lr_patch_gen.out.next()
            high_tensor[i, :, :, :] = patch_high
            low_tensor[i, :, :, :] = patch_low
        
        low_tensor /= self._std
        low_tensor -= self._mean
        high_tensor /= self._std
        high_tensor -= self._mean
        
        return low_tensor, high_tensor

    @property
    def n_files(self):
        return self._n_file
    
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

class TestImageHolder(object):
    def __init__(self,
                 tensor,
                 patch_shape, strides,
                 valid_shape, valid_offset,
                 down_sample_ratio,
                 batch_size,
                 mean=1,
                 std=128):
        super(ImageTester, self).__init__()        
        self._patch_shape = patch_shape
        self._strides = strides
        self._valid_shape = valid_shape
        self._valid_offset = valid_offset    
        self._ratio = down_sample_ratio    
        self._tensor = tensor
        self._patches = xlearn.utils.tensor.patch_generator_tensor(self._tensor,
                                                                   self._patch_shape,
                                                                   self._strides)
        self._n_patch = len(self._patches)
        self._infer = []        
        self._oid = 0
        self._iid = 0

    def next_batch(self, batch_size):        
        low_shape = [batch_size, self._patch_shape[0], self._patch_shape[1], 1]                                                    
        low_tensor = np.zeros(low_shape) 
        for i in xrange(batch_size):
            if self._oid < self._n_patch:                        
                low_tensor[i, :, :, :] = self._patches[self._oid]
            else:
                low_tensor[i, :, :, :] = np.zeros([self._patch_shape[0],
                                                   self._patch_shape[1], 1])
            self._oid += 1                    
        low_tensor /= self._std
        low_tensor -= self._mean
        return low_tensor
    
    def append_infer(self, infer_list):
        for infer in infer_list:
            infer += self._mean
            infer *= self._std
            self._infer.append(infer)
        if len(self._infer) > self._n_patch:
            self._infer = self._infer[:self._n_patch]

    def low_image(self):
        return self._tensor[0, :, :, 0]

    def recon(self):
        tensor_shape = [1, self._tensor.shape[0]*self._ratio, self._tensor.shape[1]*self._ratio, 1]
        h_patch_shape = [self._patch_shape[0]*self._ratio, self._patch_shape*self._ratio]
        h_strides = [h_patch_shape[0]-self._patch_shape[0]+self._strides[0],
                     h_patch_shape[1]-self._patch_shape[1]+self._strides[1]]
        self._recon_tensor = xlearn.utils.tensor.patches_recon_tensor(self._infer,
                                                                      tensor_shape,
                                                                      h_patch_shape,
                                                                      h_strides,
                                                                      self._valid_shape,
                                                                      self._valid_offset)
        return self._recon_tensor                                                                 

    @property
    def n_patch(self):
        return self._n_patch
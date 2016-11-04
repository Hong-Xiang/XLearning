#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2016-11-03 10:20:52

@author: HongXiang

General image inputer
"""
import os
from six.moves import xrange
import xlearn.utils.xpipes as xpipe
import xlearn.utils.dataset as xdata

class DataSet(object):
    def __init__(self, path, preix, batch_size, n_patch_per_file=1, datatype='test'):
        self._path = os.path.abspath(path)
        self._batch_size = batch_size
        self._data_type = datatype
        self._n_patch = n_patch_per_file
        if self._data_type == 'test':
            preader = xpipe.PipeFileReader(path, prefix, random_shuffle=False)
        if self._data_type == 'train':
            preader = xpipe.PipeFileReader(path, prefix, random_shuffle=True)
        ptensor = xpipe.PipeTensorFormater(preader, mindim=4)
        
        self._
        
    def next_batch():
        for i in xrange() 

    @property
    def n_files(self):
        return self._n_file

    @property
    def files_high(self):
        return self._filename_h

    @property
    def files_low(self):
        return self._filename_l

    @property
    def height(self):
        return self._patch_shape[0]

    @property
    def width(self):
        return self._patch_shape[1]

    @property
    def epoch(self):
        """current epoch"""
        return self._epoch

    @property
    def buffer_high(self):
        return self._high_res_buffer

#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2016-11-03 10:20:52

@author: HongXiang

MNIST inputer
"""
import os
import numpy as np
from six.moves import xrange
import xlearn.utils.xpipes as xpipe
import xlearn.utils.dataset as xdata
import random

class DataSet(object):
    def __init__(self, data_file, label_file, is_shuffle=True, batch_size=None):
        self._data = np.array(np.load(data_file))
        self._label = np.array(np.load(label_file))
        self._batch_size = batch_size
        self._height = self._data.shape[1]
        self._width = self._data.shape[2]
        self._num_example = self._data.shape[0]
        self._ids = list(xrange(self._num_example))        
        self._cid = 0
        self._is_shuffle = is_shuffle
        self._epoch = 0        
        if self._is_shuffle:
            random.shuffle(self._ids)

    def next_batch(self, batch_size=None):
        if batch_size is None:
            sz = self._batch_size
        else:
            sz = batch_size
        if sz is None:
            raise TypeError('No batchsize information.')
        if sz > self._num_example:
            raise ValueError('Batch size {0} is larger than dataset size {1}.'.format(sz, self._num_example))

        data = np.zeros([sz, self._height, self._width, 1])
        label = np.zeros([sz, 10])
        if self._cid + sz >= self._num_example:
            self._cid = 0
            if self._is_shuffle:
                random.shuffle(self._ids)
            self._epoch += 1
        for i in xrange(sz):
            ids = self._ids[self._cid]
            data[i, :, :, 0] = self._data[ids, :, :]
            label[i, :] = self._label[ids, :]
            self._cid += 1
        return data, label

    @property
    def epoch(self):
        return self._epoch
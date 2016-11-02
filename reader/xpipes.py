#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2016-11-02 15:13:00

@author: HongXiang

Processing pipeline models.
    Input image (loading, etc)
    RGB 2 gray pipes

"""

from __future__ import absolute_import, division, print_function
import random, os
from six.moves import xrange
import xlearning.utils


class XPipe(object):
    def __init__(self):
        self._is_start = None
        self._is_end = None
        self._branches = []
        self._name = 'XPipe'

    def _pump(self):
        if self._is_start == False:
            raise TypeError("Can't call pump for non-start pipes.")        

    def attach(self, pipe):
        if self._is_start == True:
            raise TypeError("Can't attach to an start pipe.")
        if pipe._is_end:
            raise TypeError("Can't attach an end pipe.")
        self._branches.append(pipe)

    def _merge(self, results):
        if len(results) == 1:
            return results[0]

    def next(self):
        results = []
        if self._is_start:
            yield self.pump()
        else:
            for pipe in self._branches:
                results.append(pipe.next())
            yield self.merge(results)

class PipeImageReader(XPipe):
    def __init__(self, folder, 
                 prefix, suffix='', 
                 ids=None, random_shuffle=False):
        super(ImageReader, self).__init__()
        self._path = os.path.abspath(folder)
        self._is_start = True
        self._name = 'ImageReader'
        self._ids = ids
        self._nfiles = len(self._ids)
        self._is_random = random_shuffle
        self._epoch = 0
        self._cid = 0
        self._id_shuffle = []
        self._new_epoch()

    def _new_epoch(self):
        self._epoch += 1
        self._cid = 0        
        self._id_shuffle = list(xrange(self._nfiles))
        if self._is_random:
            random.shuffle(self._id_shuffle)

    def _pump(_self):
        idnow = self._ids[self._id_shuffle[self._cid]]
        
        filename = xlearning.utils.dataset.form_file_name(self._prefix,
                                                          idnow,
                                                          self._suffix)
        fullname = os.path.join(self._path, filename)
        img = np.load(filename)
        yield img
        self._cid += 1
        if self._cid > self._nfiles:
            self._new_epoch()

        

    
    
        
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
import random, os, copy
from six.moves import xrange
import xlearn.utils
import numpy as np


class Pipe(object):
    """
    Base class of all pipes.
    """
    def __init__(self):
        self._is_start = None
        self._is_end = None
        self._branches = []        
        self._name = 'Pipe'
        self._seal = False
    
    def close():
        self._seal = True
    def open():
        self._seal = False

    def _pump(self):
        if self._is_start == False:
            raise TypeError("Can't call pump for non-start pipes.")        

    def attach(self, pipe):
        if self._is_start == True:
            raise TypeError("Can't attach to an start pipe.")
        if pipe._is_end:
            raise TypeError("Can't attach an end pipe.")
        self._branches.append(pipe)

    def _process(self, results):
        if len(results) == 1:
            return results[0]
    
    def _seal_check(self):
        if self._seal:
            raise TypeError('Seal check failed.')

    def output(self):
        results = []
        self._seal_check()
        if self._is_start:
            yield self._pump()
        else:
            for pipe in self._branches:
                results.append(pipe.output().next())
            yield self._process(results)

class PipeSingleInput(Pipe):
    """
    Base class of pipes with only one input.
    """
    def __init__(self, input_pipe):
        super(PipeSingleInput, self).__init__()
        self._is_start = False
        self._branches.append(input_pipe)
    
    def attach(self, pipe):
        raise TypeError('Attach to PipeSingleInput is not allowed.')

    def _father(self):
        return self._branches[0]


class PipeFileReader(Pipe):
    """
    .npy file reader.
    
    *Starter*

    """
    def __init__(self, folder, 
                 prefix, suffix='npy', 
                 ids=None, random_shuffle=False):
        super(PipeFileReader, self).__init__()
        self._path = os.path.abspath(folder)        
        self._is_start = True
        self._name = 'ImageReader'
        self._prefix = prefix
        self._suffix = suffix
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

    def _pump(self):
        idnow = self._ids[self._id_shuffle[self._cid]]
        
        filename = xlearn.utils.dataset.form_file_name(self._prefix,
                                                          idnow,
                                                          self._suffix)
        fullname = os.path.join(self._path, filename)
        data = np.load(filename)        
        self._cid += 1
        if self._cid == self._nfiles:
            self._new_epoch()
        return data

class PipeImageFormater(Pipe):
    def __init__(self, prepipe):
        super(PipeImageFormater, self).__init__()
        self._is_start = False
        self._branches.append(prepipe)

    def _process(self, results):
        if len(results) > 1:
            raise ValueError('PipeImageFormater only accepts one result per process.')
        data = results[0]
        if not isinstance(data, np.ndarray):
            raise TypeError('PipeImageFormater only accepts numpy.array like objects.')
        if len(data.shape) == 2:
            return data
        if len(data.shape) == 3 and data.shape[2] == 3:
            return data
        else:
            raise TypeError('PipeImageFormater only accepts H*W or H*W*3 shape arrays.')

class PipeImageGrayer(Pipe):
    def __init__(self, imagepipe):        
        super(PipeImageGrayer, self).__init__()
        self._is_start = False        
        self._branches.append(imagepipe)
            
    def _process(self, results):
        if len(results) > 1:
            raise ValueError('PipeImageGrayer only accepts one result per process.')
        img = results[0]
        gray = xlearn.utils.image.rgb2gray(img)
        return gray 
       
class PipeRepeater(Pipe):
    def __init__(self, pipe, repeat_time=1):
        super(PipeRepeater, self).__init__()
        self._is_start = False
        self._branches.append(pipe)
        self._buffer = []
        self._repeat_time = repeat_time

    def _clear_buffer(self):
        self._buffer = []
    
    def output(self):        
        self._seal_check()                             
        yield self._process()

    def _process(self):
        results = []
        for i in xrange(self._repeat_time):
            results.append(self._branches[0].output().next())  
        return results
    
class PipeConstPumper(Pipe):
    def __init__(self, char):
        super(PipeConstPumper, self).__init__()
        self._is_start = True
        self._char = char
    def _pump(self):
        return copy.deepcopy(self._char)

class PipeListExtender(Pipe):
    def __init__(self, pipes):
        super(PipeListExtender, self).__init__()
        self._is_start = False
        self._branches = pipes
        
    def _process(self, results):        
        result = []
        for i in xrange(len(results)):
            result.extend(results[i])
        return result

class PipeTensorFormater(Pipe):
    def __init__(self, pipes=[], mindim=None):
        super(PipeTensorFormater, self).__init__()
        self._is_start = False
        if not isinstance(pipes, list):
            raise TypeError('PipeTensorFormater only accepts list of pipes')       
        self._branches = pipes
        self._mindim = mindim
    
    def _process(self, results):
        results_refine = xlearn.utils.tensor.unpack_tensor_ndlist(results)
        nresult = len(results_refine)
        res_tensor = xlearn.utils.tensor.merge_tensor_list(results_refine) 
        if self._mindim is not None and len(res_tensor.shape) < self._mindim:
            toadd = self._mindim -  len(res_tensor.shape)
            sztmp = []
            for i in xrange(toadd):
                sztmp.append(1)
            newshape = sztmp + list(res_tensor.shape)
            res_tensor = np.reshape(res_tensor, newshape)
        return res_tensor

class PipePatchGenerator(Pipe):
    def __init__(self, tensorpipe, shape, strides, random_gen=False, n_patches=None):
        super(PipePatchGenerator, self).__init__()
        self._shape = shape
        self._strides = strides
        self._random = random_gen
        self._n_patches = n_patches
        self._branches.append(tensorpipe)
    
    def _process(self, result):        
        if len(result) > 1:
            raise TypeError('Only one input is support, given %d.'%len(result))
        result = result[0]        
        if not isinstance(result, np.ndarray):
            raise TypeError('Invalid input, required ' 
                            + str(np.ndarray) 
                            + ' get ' + str(type(result)) 
                            + '.')
        output_result = []
        patch_gen = xlearn.utils.tensor.patch_generator_tensor(result, 
                                                               self._shape,
                                                               self._strides,
                                                               self._n_patches,
                                                               self._random)
        for patches in patch_gen:            
            output_result.append(patches)
        return output_result


class PipeCopyer(PipeSingleInput):
    """
    Copy and buffer result from input pipe.

    *Mid*

    Can used to broadcast results.
    """
    def __init__(self, pipe, copy_number=1):
        super(PipeCopyer, self).__init__(pipe)        
        self._copy_number = copy_number
        if self._copy_number == 0:
            raise ValueError('Maximum copy number must be positive, input %d.'%copy_number)
        self._copyed = 0
        self._buffer = None
        self._branches.append(pipe)

    def _new_buffer(self):
        self._buffer = self._father().output().next()
        self._copyed = 0

    def output(self):
        self._seal_check()
        if self._buffer is None:
            self._new_buffer()
        if self._copyed == self._copy_number:
            self._new_buffer()
        self._copyed += 1
        yield self._buffer

        


    
    
        
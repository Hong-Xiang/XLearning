#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2016-11-02 15:13:00


Processing pipeline models.
    Input image (loading, etc)
    RGB 2 gray pipes
"""
from __future__ import absolute_import, division, print_function
__author__ = 'Hong Xiang'
import random
import os
import copy
from six.moves import xrange
import xlearn.utils.tensor
import xlearn.utils.image
import xlearn.utils.general
import numpy as np
import scipy.misc
import re


class Pipe(object):
    """
    Base class of all pipes.
    It can also work as a merge pipe for multiple inputs. Returning a list.
    
    Attributions
    ___
    positions
        - start
        - mid
        - end
    pump 
        - *NO* implemention.
    input
        - None
        - Pipe
        - list of Pipes 
    output
        - list
    
    List of Pipes
    ---
    .. [1] Not implemented    
    """
    def __init__(self, input_=None, name='Pipe', is_start=None, is_seal=False):
        self._name = name
        self._branches = []
        if input_ is None:
            pass
        elif hasattr(input_, '__iter__'):        
            for input_pipe in input_:
                if not isinstance(input_pipe, Pipe):
                    raise TypeError('Required {0} given {1}.'.format(Pipe, type(input_pipe)))
                self._branches.append(input_pipe)
        else:
            if not isinstance(input_, Pipe):
                    raise TypeError('Required {0} given {1}.'.format(Pipe, type(input_)))
            self._branches.append(input_)
        self._is_start = is_start             
        self._is_seal = is_seal
        

    def close():
        self._is_seal = True

    def open():
        self._is_seal = False

    def _pump(self):
        raise RuntimeWarning('Pump of Pipe called, but no implementation yet.')
        return None

    def attach(self, pipe):
        if self.is_start:
            raise TypeError("Can't attach to an start pipe.")        
        self._branches.append(pipe)
        self._is_start = False

    def _pump(self):
        return None

    def _gather(self):
        if self.is_start:
            return self._pump()
        else:    
            results = []
            for pipe in self._branches:                                
                    results.append(pipe.out.next())
            return results

    def _process(self):
        return self._gather()
    
    @property
    def out(self):        
        if self.is_seal:
            yield None       
        else:
            yield self._process()
    
    @property
    def name(self):
        return self._name
    
    @property
    def is_start(self):
        return self._is_start

    @property
    def is_end(self):
        return self._is_end
    
    @property
    def is_seal(self):
        return self._is_seal

    @property
    def branches(self):
        return self._branches

class SingleInput(Pipe):
    """
    Base class of pipes with only one input.
    """
    def __init__(self, input_, name='SingleInput', is_seal=False):
        super(SingleInput, self).__init__(input_,
                                          name=name,
                                          is_start=False,                                        
                                          is_seal=is_seal)
        
    def attach(self, input_):
        raise TypeError('Attach to {0} is not allowed.'.format(SingleInput))

    @property
    def _father(self):
        return self._branches[0]
    
class Buffer(Pipe):
    """
    Buffer type pipe. can buffer a result.
    
    Basically if it was linked to to a pipe which generates output with list,
    Buffer will store its result and output one element of the list per time.
    """
    def __init__(self, input_=None, name='Buffer', is_start=None, is_seal=False):
        super(Buffer, self).__init__(input_, name=name, is_start=is_start, is_seal=is_seal)
        self._buffer = None
        self._state = 0
        self._max_state = None
        
    def _new_buffer(self):
        if self._is_start:
            self._buffer = [self._pump()]                
        else:
            self._buffer = self._gather()        
        self._buffer = xlearn.utils.general.unpack_list(self._buffer)
        if not hasattr(self._buffer, '__iter__'):
                raise TypeError('Buffer requires inputs/pumps with __iter__ attr.')
        self._max_state = len(self._buffer)
        self._state = 0
        
    def _process(self):
        if self._buffer is None:
            self._new_buffer()
        if self._state == self._max_state:
            self._new_buffer()
        output = self._buffer[self._state] 
        self._state += 1
        return output

class Counter(Pipe):
    def __init__(self, name='Counter', is_seal=False):
        super(Counter, self).__init__(input_=None, name=name, is_start=True, is_seal=is_seal)        
        self._state = 0        
        
    def _pump(self):
        self._state += 1
        return self._state

    def reset(self):
        self._state = 0
    
    def count(self):
        return self._state


class ConstPumper(Buffer):
    """
    General input of pipe system.    
    """
    def __init__(self, const_item, name='ConstPumper', is_seal=False):
        super(ConstPumper, self).__init__(name=name, is_start=True, is_seal=is_seal)         
        self._const = copy.deepcopy(const_item)        
    
    def _pump(self):        
        return self._const
    
    def set(const_item):
        self._const = const_item
        
    @property
    def const_item(self):
        return self._const

class Repeater(SingleInput):
    def __init__(self, input_, repeat_time=1, name='Repeater', is_seal=False):
        super(Repeater, self).__init__(input_, name=name, is_seal=is_seal)        
        self._buffer = []
        self._repeat_time = repeat_time
        
    def _process(self):        
        for i in xrange(self._repeat_time):
            input_ = self._gather()
            if input_ is None:
                return None
            else:
                return [input_]
        
    

class NPYReader(Pipe):
    """
    .npy file reader.
    
    [start]
    <output> tensor from .npy file.
    """
    def __init__(self, folder,
                 prefix,
                 ids=None,
                 random_shuffle=False,
                 name='NPYReader',
                
                 is_seal=False):
        super(NPYReader, self).__init__(is_start=True,
                                        name=name,
                                    
                                        is_seal=is_seal)
        self._path = os.path.abspath(folder)        
        self._prefix = prefix
        self._suffix = 'npy'
        list_all = os.listdir(self._path)
        list_all.sort()   
        self._file_names = []
        if ids is None:
            p = re.compile(self._prefix+'\d+'+'.'+self._suffix)
            for file in list_all:
                if p.match(file):
                    self._file_names.append(file)
        else:            
            for id_ in ids:            
                filename = xlearn.utils.dataset.form_file_name(self._prefix,
                                                               id_,
                                                               self._suffix)
                if filename in list_all:
                    self._file_names.append(filename)        
        self._nfiles = len(self._file_names)
        self._is_random = random_shuffle
        self._epoch = 0
        self._cid = 0
        self._id_shuffle = []
        self._new_epoch()
        self._shape = None

    def _new_epoch(self):
        self._epoch += 1
        self._cid = 0        
        self._id_shuffle = list(xrange(self._nfiles))
        if self._is_random:
            random.shuffle(self._id_shuffle)

    def _pump(self):
        filename = self._file_names[self._id_shuffle[self._cid]]        
        fullname = os.path.join(self._path, filename)
        data = np.array(np.load(fullname))
        self._shape = data.shape
        self._cid += 1
        if self._cid == self._nfiles:
            self._new_epoch()
        return data
    
    @property
    def n_files(self):
        return self._nfiles

    @property
    def last_shape(self):        
        return self._shape

    @property
    def epoch(self):
        return self._epoch

class ImageFormater(SingleInput):
    """
    Convert tensor input format which is plotable.
    """
    def __init__(self, input_, is_uint8=False, offset=3, name='ImageFormater', is_seal=False):
        super(ImageFormater, self).__init__(input_, name=name, is_seal=is_seal)                
        self._is_uint8=is_uint8
        self._offset = offset

    def _process(self):
        img_maybe = self._gather()
        if img_maybe is None:
            return None
        img_maybe = img_maybe[0]        
        if not isinstance(img_maybe, np.ndarray):
            raise TypeError('ImageFormater only accepts {0}, given {1}'.format(np.ndarray, type(img_maybe)))
        img_type = xlearn.utils.tensor.image_type(img_maybe)
        output = img_maybe
        if img_type is 'gray' or img_type is 'RGB':
            output = img_maybe
        if img_type is 'gray1':
            output = img_maybe[:, :, 0]
        if img_type is '1gray1':
            output = img_maybe[0, :, :, 0]
        if img_type is 'Ngray':
            if img_maybe.shape[0] == 1:
                output = img_maybe[0, :, :]
            else:
                output = xlearn.utils.tensor.multi_image2large_image(img_maybe, offset=self._offset)
        if img_type is 'NRGB':
            if img_maybe.shape[0] == 1:
                output = img_maybe[0, :, :, :]
            else:
                output = xlearn.utils.tensor.multi_image2large_image(img_maybe, offset=self._offset)
        if self._is_uint8:
            output = np.uint8(output)
        return output

    def set_uint8(self, flag):
        self._is_uint8 = flag
    
    def set_offset(self, offset):
        self._offset = offset

class ImageGrayer(SingleInput):
    def __init__(self, input_, name='ImageGrayer', is_seal=False):       
        super(ImageGrayer, self).__init__(input_, name=name, is_seal=is_seal)        

    def _process(self):
        image = self._gather()
        if image is None:
            return None
        image = image[0]      
        gray = xlearn.utils.image.rgb2gray(image)
        return gray 
       



class TensorFormater(SingleInput):
    def __init__(self, input_, squeeze=True, newshape=None, auto_shape=False, name='TensorFormater', is_seal=False):
        super(TensorFormater, self).__init__(input_, name=name, is_seal=is_seal)
        self._shape = newshape        
        self._auto_shape = auto_shape
        self._squeeze = squeeze

    def _process(self):
        tensor_list = self._gather()
        tensor_list = tensor_list[0]               
        if tensor_list is None:
            return None        
        output = xlearn.utils.tensor.merge_tensor_list(tensor_list, squeeze=self._squeeze)
        preshape = output.shape             
        if self._shape is not None:
            newshape = self._shape
        elif self._auto_shape:                        
            if len(preshape) == 3:
                newshape = [1, 1, 1, 1]
                newshape[-len(preshape):]=preshape
            elif len(preshape) == 2:
                newshape = [1, 1, 1, 1]
                newshape[1:3] = preshape
            else:
                newshape = preshape
        else:
            newshape = preshape        
        output = np.reshape(output, newshape)
        return output

class PatchGenerator(SingleInput):
    def __init__(self, input_,
                 shape, strides, random_gen=False, n_patches=None,
                 name='PatchGenerator', is_seal=False):
        super(PatchGenerator, self).__init__(input_, name=name, is_seal=is_seal)
        self._shape = shape
        self._strides = strides
        self._random = random_gen
        self._n_patches = n_patches        
    
    def _gather_with_check(self):
        tensor = self._gather()
        if tensor is None:
            return None
        tensor = tensor[0]
        if not isinstance(tensor, np.ndarray):
            raise TypeError('Input required {0}, got {1}.'.format(np.ndarray, type(tensor)))
        if tensor.shape[1] < self._shape[0] or tensor.shape[2] < self._shape[1]:
            return self._gather_with_check()
        else:
            return tensor
                
    def _process(self):
        tensor = self._gather_with_check()                                            
        output = []
        patch_gen = xlearn.utils.tensor.patch_generator_tensor(tensor,
                                                               self._shape,
                                                               self._strides,
                                                               self._n_patches,
                                                               self._random)
        for patches in patch_gen:
            output.append(patches)
        return output

class PatchMerger(SingleInput):
    def __init__(self, input_, 
                 tensor_shape, patch_shape, strides,
                 valid_shape, valid_offset,
                 name="PatchMerger", is_end=None, is_seal=False):
        super(PatchMerger, self).__init__(input_, is_seal=is_seal)
        self._tensor_shape = tensor_shape        
        self._patch_shape = patch_shape
        self._strides = strides
        self._valid_shape = valid_shape
        self._valid_offset = valid_offset        
    
    def _process(self):
        input_ = self._gather()
        if input_ is None:
            return None
        input_ = input_[0]
        output_ = xlearn.utils.tensor.patches_recon_tensor(input_,
                                                           self._tensor_shape,
                                                           self._patch_shape,
                                                           self._strides,
                                                           self._valid_shape,
                                                           self._valid_offset)
        return output_

class Copyer(Buffer):
    """
    Copy and buffer result from input pipe.

    *Mid*

    Can used to broadcast results.
    """
    def __init__(self, input_, copy_number=1, name="Copyer", is_start=None, is_seal=False):
        super(Copyer, self).__init__(input_, name=name, is_start=False, is_seal=is_seal)         
        self._copy_number = copy_number
        if self._copy_number == 0:
            raise ValueError("Maximum copy number can't be zero.'")
        self._max_state = self._copy_number        

    def _process(self):
        if self._buffer is None:
            self._new_buffer()
            self._max_state = self._copy_number
        if self._state == self._max_state:
            self._new_buffer()
            self._max_state = self._copy_number
        output = self._buffer[0]
        self._state += 1
        return output

class Proj2Sino(Buffer):
    def __init__(self, input_, name='Proj2Sino', is_start=False, is_seal=False):
        super(Proj2Sino, self).__init__(input_, name=name, is_start=is_start, is_seal=is_seal)

    def _new_buffer(self):        
        proj_data = self._gather()
        if proj_data is None:
            self._buffer = None
            return None
        proj_data = proj_data[0]
        height = proj_data.shape[0]
        width = proj_data.shape[1]
        nangle = proj_data.shape[2]
        sinograms = []
        for iz in xrange(height):
            sinogram = np.zeros([width, nangle])
            for iwidth in range(width):
                for iangle in range(nangle):
                    sinogram[iwidth, iangle] = proj_data[iz, iwidth, iangle]
            sinograms.append(sinogram)
        self._buffer = sinograms
        self._max_state = height
        self._state = 0

class DownSampler(SingleInput):
    def __init__(self, input_, ratio=1, method='mean', name='DownSampler', padding=False, is_seal=False):
        super(DownSampler, self).__init__(input_, name=name, is_seal=is_seal)
        self._ratio = ratio
        self._method = method
        self._padding = padding
    def _process(self):
        """
        --------++-----------++---------++----------++
        ratio   ||     1     ||    2    ||     3    ||
        --------++-----------++---------++----------++
        down    ||    +-+    ||  +-+-+  ||  +-+-+-+ ||
                ||    |x|    ||  |x| |  ||  | | | | ||
                ||    +-+    ||  +-+-+  ||  +-+-+-+ ||
                ||           ||  | | |  ||  | |x| | ||
                ||           ||  +-+-+  ||  +-+-+-+ ||
                ||           ||         ||  | | | | ||
                ||           ||         ||  +-+-+-+ ||
        --------++-----------++---------++----------++
        """
        input_ = self._gather()
        input_ = input_[0]
        if not isinstance(input_, np.ndarray):
            raise TypeError('Input of {0} is required, get {1}'.format(np.ndarray), type(input_))
        if len(input_.shape) != 4:
            raise TypeError('Tensor of 4 D is required, get %d D.'%len(input_.shape))

        ratio = self._ratio  
        input_height = input_.shape[1]
        input_width = input_.shape[2]
        if self._padding:
            output_height = int(np.ceil(input_height/ratio))
            output_width = int(np.ceil(input_width/ratio))
        else:
            output_height = int(input_height/ratio)
            output_width = int(input_width/ratio)
        n_image = input_.shape[0]
        n_channel = input_.shape[3]
        output_shape = [n_image, output_height, output_width, n_channel]            
        output = np.zeros(output_shape)
        
        for iy in xrange(output_height):
            for ix in xrange(output_width):
                for ii in xrange(n_image):
                    for ic in xrange(n_channel):
                        output[ii, iy, ix, ic] = 0
                        if self._method == 'mean':
                            for stepy in xrange(ratio):
                                for stepx in xrange(ratio):
                                    idy = iy*ratio + stepy
                                    idx = ix*ratio + stepx
                                    if idy >= input_height or idx >= input_width:
                                        picked = 0
                                    else:
                                        picked = input_[ii, idy, idx, ic]
                                    output[ii, iy, ix, ic] += picked
                            output[ii, iy, ix, ic] /= (ratio*ratio)
                        if self._method == 'fixed':
                            idy = iy*ratio + int((ratio-1)/2)
                            idx = ix*ratio + int((ratio-1)/2)
                            if idy >= input_height or idx >= input_width:
                                picked = 0
                            else:
                                picked = input_[ii, idy, idx, ic]
                            output[ii, iy, ix, ic] = picked 
        return output
        

        


        


    
    
        
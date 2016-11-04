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
import xlearn.utils
import numpy as np
import scipy.misc


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
        if self._max_state is None:
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
    def __init__(self, const_item, name='ConstPumper', is_seal=False):
        super(ConstPumper, self).__init__(name=name, is_start=True, is_seal=is_seal)         
        self._const = copy.deepcopy(const_item)        
    
    def _pump(self):        
        return self._const
    
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
        self._ids = ids
        self._nfiles = len(self._ids)
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
        idnow = self._ids[self._id_shuffle[self._cid]]        
        filename = xlearn.utils.dataset.form_file_name(self._prefix,
                                                          idnow,
                                                          self._suffix)
        fullname = os.path.join(self._path, filename)
        data = np.array(np.load(filename))
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
                output = img_maybe[0, :, :]
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
    
    def _process(self):        
        tensor = self._gather()
        if tensor is None:
            return None
        tensor = tensor[0]
        if not isinstance(tensor, np.ndarray):
            raise TypeError('Input required {0}, got {1}.'.format(np.ndarray, type(tensor)))                            
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

# class PipeCopyer(PipeSingleInput):
#     """
#     Copy and buffer result from input pipe.

#     *Mid*

#     Can used to broadcast results.
#     """
#     def __init__(self, pipe, copy_number=1):
#         super(PipeCopyer, self).__init__(pipe)        
#         self._copy_number = copy_number
#         if self._copy_number == 0:
#             raise ValueError('Maximum copy number must be positive, input %d.'%copy_number)
#         self._copyed = 0
#         self._buffer = None
#         self._branches.append(pipe)

#     def _new_buffer(self):
#         self._buffer = self._father.output().next()
#         self._copyed = 0

#     def output(self):
#         self._is_seal_check()
#         if self._buffer is None:
#             self._new_buffer()
#         if self._copyed == self._copy_number:
#             self._new_buffer()
#         self._copyed += 1
#         yield self._buffer

# class DownSampler(PipeSingleInput):
#     def __init__(self, pipe, ratio=1, method='nearest'):
#         super(DownSampler, self).__init__(pipe)
#         self._ratio = float(ratio)
#         self._method = method

#     def _new_shape(self, sz_old):
#         sz = list(sz_old)
#         sz[0] = np.ceil(sz[0]/self._ratio)
#         sz[1] = np.ceil(sz[1]/self._ratio)
#         return sz

#     def _down_sample(self, img):
#         sz = self._new_shape(img.shape)
#         return scipy.misc.imresize(img, sz, interp=method)

#     def _process(self, tensor_in):
#         if xlearn.utils.tensor.image_type(tensor_in) == 'gray':
#             return self._down_sample(tensor_in)
#         if xlearn.utils.tensor.image_type(tensor_in) == 'RGB':
#             sz = self._new_shape(tensor_in[:2])
#             tensor_out = np.zeros(sz+[3])
#             for i in xrange(3):
#                 tensor_out[:, :, i] = self._down_sample(tensor_in[:, :, i])
#         if xlearn.utils.tensor.image_type(tensor_in) == 'Ngray1':
#             sz = self._new_shape(tensor_in[1:3])
#             nimg = tensor_in.shape[0]
#             tensor_out = np.zeros([nimg]+sz+[1])
#             for i in xrange(tensor_in.shape[0]):
#                 tensor_out[i, :, :, 0] = self._down_sample(tensor_in[i, :, :, 0])
        

        


        


    
    
        
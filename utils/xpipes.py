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
import re
import collections
import time
import numpy as np
from six.moves import xrange
import scipy.misc

import xlearn.utils.tensor
import xlearn.utils.image
import xlearn.utils.general


class Pipe(object):
    """
    Base class of all pipes.
    It can also work as a merge pipe for multiple inputs, in this case, it will return a list.

    Attributions
    ___
    pump

    input
        - None
        - Pipe
        - list of Pipes
    output
        - None
        - list of objects
    """

    def __init__(self, input_=None, name='Pipe', is_start=None, is_seal=False):
        self._name = name
        if input_ is None:
            input_ = []
        if not hasattr(input_, '__iter__'):
            input_ = [input_]
        self._n_inputs = len(input_)
        if is_start is False and self.n_inputs == 0:
            raise TypeError("None start pipe without input pipe.")
        self._is_start = len(input_) == 0
        self._branches = []
        for pipe in input_:
            self._attach(pipe)        
        self._check_input_type()

        self._is_seal = is_seal
        self._output_type = (object, list)

    def _check_input_type(self):
        for id_pipe in xrange(self.n_inputs):
            if self._branches[id_pipe].output_type != object:
                raise TypeError("Output Type mismatch of %d-th input."%id_pipe)

    def close(self):
        self._is_seal = True

    def open(self):
        self._is_seal = False

    def _pump(self):
        return None

    def _attach(self, pipe):
        if self.is_start:
            raise TypeError("Can't attach to an start pipe.")
        if not isinstance(pipe, Pipe):
            raise TypeError('Required {0} given {1}.'.format(Pipe,
                                                             type(pipe)))
        self._branches.append(pipe)
        self._is_start = False

    def _gather(self):
        if self.is_start:
            output = self._pump()
            return output
        else:
            results = []
            for pipe in self._branches:
                results.append(pipe.out.next())
            return results

    def _process(self):
        return self._gather()

    @property
    def output_type(self):
        return self._output_type

    @property
    def out(self):
        while True:
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
    def is_seal(self):
        return self._is_seal

    @property
    def branches(self):
        return self._branches

    @property
    def n_inputs(self):
        return self._n_inputs


class SingleInput(Pipe):
    """
    Base class of pipes with only one input.
    """
    def __init__(self, input_, name='SingleInput', is_seal=False):
        if not isinstance(input_, Pipe):
            raise TypeError('Required {0} given {1}.'.format(Pipe,
                                                             type(input_)))
        super(SingleInput, self).__init__(input_,
                                          name=name,
                                          is_start=False,
                                          is_seal=is_seal)

    @property
    def _father(self):
        return self._branches[0]

class Counter(Pipe):
    def __init__(self, max_state=None, is_frozen=False, name='Counter', is_seal=False):
        super(Counter, self).__init__(input_=None,
                                      name=name, is_start=True, is_seal=is_seal)
        self._state = 0
        self._max_state = max_state
        self._is_frozen = is_frozen

    def _pump(self):
        if not self.is_frozen:
            if self._state == self._max_state or self._max_state == 0:
                raise StopIteration()
        output = self._state
        if not self.is_frozen:
            self._state += 1
        return output

    def reset(self):
        self._state = 0

    @property
    def state(self):
        return self._state
        
    @property
    def max_state(self):
        return self._max_state

    @max_state.setter
    def max_state(self, value):
        self._max_state = value

    @property
    def is_frozen(self):
        return self._is_frozen

    def is_end(self):
        if self._max_state is None:
            return False
        if self._state < self._max_state:
            return False
        return True

class Buffer(Pipe):
    """
    Buffer type pipe. can buffer a result list.

    If Buffer pipe was linked to a pipe which generates output with list, Buffer
    pipe will store its result and output one element of the list per time.
    """
    def __init__(self, input_=None, fixed_max_state=False,
                 name='Buffer', is_start=None, is_seal=False):
        super(Buffer, self).__init__(input_, name=name,
                                     is_start=is_start, is_seal=is_seal)
        self._buffer = None
        self._counter = Counter()
        self._is_fixed_max_state = fixed_max_state

    def _new_buffer(self):
        self._buffer = self._gather()
        if not self._is_fixed_max_state:
            self._counter.max_state = len(self._buffer)
        self._counter.reset()

    def _process(self):
        if self._buffer is None:
            self._new_buffer()
        if self._counter.is_end():
            self._new_buffer()
        output = self._buffer[next(self._counter.out)]
        return output

    @property
    def counter(self):
        return self._counter

class Inputer(Buffer):
    def __init__(self, item=None, const_output=False, name='Inputer'):
        super(Inputer, self).__init__(fixed_max_state=True, name=name, is_start=True)
        self._buffer = collections.deque()
        if const_output:
            self._counter.max_state = None
        else:
            self._counter.max_state = 1
        if item is not None:
            self.insert(item)

    def insert(self, item):
        if not hasattr(item, "__iter__"):
            item = [item]
        self._buffer.extend(item)

    def _process(self):
        if len(self._buffer) == 0:
            raise StopIteration()
        next(self._counter.out)
        if self._counter.is_end():
            output = self._buffer.popleft()
            self._counter.reset()
        else:
            output = self._buffer[0]
        return output

class Copyer(Buffer):
    """
    Copy and buffer result from input pipe.
    Can used to broadcast results.
    """
    def __init__(self, input_, copy_number=1, name="Copyer", is_start=None, is_seal=False):
        super(Copyer, self).__init__(
            input_, fixed_max_state=True, name=name, is_start=False, is_seal=is_seal)
        self._counter.max_state = copy_number

    @property
    def copy_number(self):
        return self._counter.max_state

    @copy_number.setter
    def copy_number(self, value):
        self._counter.max_state = value

    def _process(self):
        if self._buffer is None:
            self._new_buffer()
        if self._counter.is_end():
            self._new_buffer()
        next(self._counter.out)
        output = self._buffer[0]
        return output

class RandomPrefix(Pipe):
    def __init__(self, prefix=None, name='RandomPrefix', is_seal=False):
        super(RandomPrefix, self).__init__(is_start=True, name=name, is_seal=is_seal)
        if prefix is None:
            self._prefix = "TMPRAND"
        else:
            self._prefix = prefix
        time_str_list = list(str(time.time()))
        time_str_list.remove('.')
        self._prefix += ''.join(time_str_list)
        self._counter = Counter()

    def _pump(self):
        output = self._prefix + "%09d"%(next(self._counter.out),)
        return output

class NPYReader(Pipe):
    """Load all *.npy file with which matches with file name like 'prefix\d9.npy' under given folder.
    <input> None
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
        self._output_type = np.ndarray
        self._path = os.path.abspath(folder)
        self._prefix = prefix
        self._suffix = 'npy'
        list_all = os.listdir(self._path)
        list_all.sort()
        self._file_names = []
        if ids is None:
            pre = re.compile(self._prefix + r'\d9' + '.' + self._suffix)
            for file in list_all:
                if pre.match(file):
                    self._file_names.append(file)
        else:
            for id_ in ids:
                filename = xlearn.utils.dataset.form_file_name(self._prefix,
                                                               id_,
                                                               self._suffix)
                if filename in list_all:
                    self._file_names.append(filename)
        self._epoch_counter = Counter()
        self._fid_counter = Counter()
        self._fid_counter.max_state = len(self._file_names)
        self._is_random = random_shuffle
        self._new_epoch()
        self._shape = None

    def _new_epoch(self):
        # TODO: Add a StopIteration raised by hitting maximum epoch number.
        next(self._epoch_counter.out)
        self._fid_counter.reset()
        if self._is_random:
            random.shuffle(self._file_names)

    def _pump(self):
        filename = self._file_names[next(self._fid_counter.out)]
        fullname = os.path.join(self._path, filename)
        data = np.array(np.load(fullname))
        self._shape = data.shape
        if self._fid_counter.is_end():
            self._new_epoch()
        return data

    @property
    def n_files(self):
        return self._fid_counter.max_state

    @property
    def last_shape(self):
        return self._shape

    @property
    def epoch(self):
        return self._epoch_counter.state

class NPYWriter(Pipe):
    """Write a series of numpy.ndarray to npy files.
    """
    def __init__(self, path, prefix, data, count, name='NPYWriter'):
        super(NPYWriter, self).__init__()
        self._branches.append(data)
        self._branches.append(count)
        self._branches[1].reset()
        self._prefix = prefix
        self._path = os.path.abspath(path)

    def _process(self):
        input_ = self._gather()
        data = input_[0]
        count = input_[1]
        filename = xlearn.utils.dataset.form_file_name(self._prefix,
                                                       count,
                                                       'npy')
        fullname = os.path.join(self._path, filename)
        np.save(fullname, data)

class ImageFormater(SingleInput):
    """
    Convert tensor input format which is plotable.
    """
    def __init__(self, input_, is_uint8=False, offset=3, name='ImageFormater', is_seal=False):
        super(ImageFormater, self).__init__(input_, name=name, is_start=False, is_seal=is_seal)
        self._is_uint8 = is_uint8
        self._offset = offset

    def _process(self):
        img_maybe = self._gather()
        if img_maybe is None:
            return None
        img_maybe = img_maybe[0]
        if not isinstance(img_maybe, np.ndarray):
            raise TypeError('ImageFormater only accepts {0}, given {1}'.format(
                np.ndarray, type(img_maybe)))
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
                output = xlearn.utils.tensor.multi_image2large_image(
                    img_maybe, offset=self._offset)
        if img_type is 'NRGB':
            if img_maybe.shape[0] == 1:
                output = img_maybe[0, :, :, :]
            else:
                output = xlearn.utils.tensor.multi_image2large_image(
                    img_maybe, offset=self._offset)
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
        if len(image) > 1:
            raise TypeError("Wrong input length.")
        image = image[0]        
        gray = xlearn.utils.image.rgb2gray(image)
        return gray


class TensorFormater(SingleInput):

    def __init__(self, input_, squeeze=True, new_shape=None, auto_shape=True,
                 name='TensorFormater', is_seal=False):
        super(TensorFormater, self).__init__(
            input_, name=name, is_seal=is_seal)
        self._shape = new_shape
        self._auto_shape = auto_shape
        self._squeeze = squeeze

    def _process(self):
        tensor_list = self._gather()
        tensor_list = tensor_list[0]
        if tensor_list is None:
            return None
        output = xlearn.utils.tensor.merge_tensor_list(
            tensor_list, squeeze=self._squeeze)

        preshape = output.shape
        if self._shape is not None:
            newshape = self._shape
        elif self._auto_shape:
            if len(preshape) == 3 and tensor_list[0].shape[0] == 1:
                newshape = [1, 1, 1, 1]
                newshape[:-1] = preshape
            elif len(preshape) == 3:
                newshape = [1, 1, 1, 1]
                newshape[-len(preshape):] = preshape
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
                 shape, strides, random_gen=False, n_patches=None, threshold=None,
                 name='PatchGenerator', is_seal=False):
        super(PatchGenerator, self).__init__(
            input_, name=name, is_seal=is_seal)
        self._shape = shape
        self._strides = strides
        self._random = random_gen
        self._n_patches = n_patches
        self._threshold = threshold
    
    def _check_input_type(self, id):

    def _gather_with_check(self):
        tensor = self._gather()
        if tensor is None:
            return None
        tensor = tensor[0]
        if not isinstance(tensor, np.ndarray):
            raise TypeError('Input required {0}, got {1}.'.format(
                np.ndarray, type(tensor)))
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
                                                               self._random,
                                                               threshold=self._threshold)
        for patches in patch_gen:
            output.append(patches)
        while len(output) == 0:
            tensor = self._gather_with_check()
            patch_gen = xlearn.utils.tensor.patch_generator_tensor(tensor,
                                                                   self._shape,
                                                                   self._strides,
                                                                   self._n_patches,
                                                                   self._random,
                                                                   threshold=self._threshold)
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




class Proj2Sino(Buffer):

    def __init__(self, input_, name='Proj2Sino', is_start=False, is_seal=False):
        super(Proj2Sino, self).__init__(
            input_, name=name, is_start=is_start, is_seal=is_seal)

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


class DownSamplerSingle(SingleInput):

    def __init__(self, input_, axis=2, ratio=1, method='mean', name='DownSampler', padding=False):
        super(DownSamplerSingle, self).__init__(input_, name=name)
        self._ratio = ratio
        self._method = method
        self._padding = padding
        self._axis = axis

    def _process(self):
        input_ = self._gather()
        input_ = input_[0]
        if not isinstance(input_, np.ndarray):
            raise TypeError('Input of {0} is required, get {1}'.format(
                np.ndarray, type(input_)))
        if len(input_.shape) != 4:
            raise TypeError('Tensor of 4 D is required, get %d D.' %
                            len(input_.shape))
        ratio = self._ratio
        axis = self._axis
        input_shape = list(input_.shape)

        if self._padding:
            new_slices = np.ceil(input_shape[axis] / ratio)
            input_shape[axis] = new_slices * ratio
            padded = np.pad(input_, input_shape, mode='constant')
        else:
            new_slices = int(input_shape[axis] / ratio)
        input_shape = list(input_shape)
        output_shape = list(input_shape)
        output_shape[axis] = new_slices

        output = np.zeros(output_shape)
        for ii in xrange(output_shape[0]):
            for ix in xrange(output_shape[2]):
                for iy in xrange(output_shape[1]):
                    for ic in xrange(output_shape[3]):
                        if self._method == 'mean':
                            for step in xrange(ratio):
                                if axis == 0:
                                    picked = input_[
                                        ii * ratio + step, iy, ix, ic]
                                if axis == 1:
                                    picked = input_[
                                        ii, iy * ratio + step, ix, ic]
                                if axis == 2:
                                    picked = input_[
                                        ii, iy, ix * ratio + step, ic]
                                if axis == 3:
                                    picked = input_[
                                        ii, iy, ix, ic * ratio + step]
                                output[ii, iy, ix, ic] += picked
                            output[ii, iy, ix, ic] /= ratio
                        if self._method == 'fixed':
                            if axis == 0:
                                ids = ii * ratio + int((ratio - 1) / 2)
                                picked = input_[ids, iy, ix, ic]
                            if axis == 1:
                                ids = iy * ratio + int((ratio - 1) / 2)
                                picked = input_[ii, ids, ix, ic]
                            if axis == 2:
                                ids = ix * ratio + int((ratio - 1) / 2)
                                picked = input_[ii, iy, ids, ic]
                            if axis == 3:
                                ids = ic * ratio + int((ratio - 1) / 2)
                                picked = input_[ii, iy, ix, ids]
                            output[ii, iy, ix, ic] = picked
        return output


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
            raise TypeError('Input of {0} is required, get {1}'.format(
                np.ndarray, type(input_)))
        if len(input_.shape) != 4:
            raise TypeError('Tensor of 4 D is required, get %d D.' %
                            len(input_.shape))

        ratio = self._ratio
        input_height = input_.shape[1]
        input_width = input_.shape[2]
        if self._padding:
            output_height = int(np.ceil(input_height / ratio))
            output_width = int(np.ceil(input_width / ratio))
        else:
            output_height = int(input_height / ratio)
            output_width = int(input_width / ratio)
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
                                    idy = iy * ratio + stepy
                                    idx = ix * ratio + stepx
                                    if idy >= input_height or idx >= input_width:
                                        picked = 0
                                    else:
                                        picked = input_[ii, idy, idx, ic]
                                    output[ii, iy, ix, ic] += picked
                            output[ii, iy, ix, ic] /= (ratio * ratio)
                        if self._method == 'fixed':
                            idy = iy * ratio + int((ratio - 1) / 2)
                            idx = ix * ratio + int((ratio - 1) / 2)
                            if idy >= input_height or idx >= input_width:
                                picked = 0
                            else:
                                picked = input_[ii, idy, idx, ic]
                            output[ii, iy, ix, ic] = picked
        return output


class TensorSlicer(SingleInput):
    """
    Slice a 4D NHW1 tensor into a list of HW tensor.
    """

    def __init__(self, input_, name='TensorSlicer'):
        super(TensorSlicer, self).__init__(input_, name)

    def _process(self):
        input_ = self._gather()
        input_ = input_[0]
        output = xlearn.utils.tensor.crop_tensor()
        
        for i in xrange(input_.shape[0]):
            output.append(input_[i, :, :, 0])
        return output


class PeriodicalPadding(SingleInput):
    """
    Padding for sinograms.

    Zero padding for y.
    Periodical padding for x.

    padding method see
    """

    def __init__(self, input_, x_padding, y_padding, period):
        super(PeriodicalPadding, self).__init__(input_)
        self._x_padding = x_padding
        self._y_padding = y_padding

    def _process(self):
        input_ = self._gather()        
        input_ = input_[0]
        if input_.shape[1] < self._period:
            raise ValueError('Input too small, width {0} smaller than period {1}.'.format(
                input_.shape[1], self._period))
        output = np.zeros(self._shape)
        height_new, width_new = self._shape[0], self._shape[1]
        height_old, width_old = input_.shape[0], input_.shape[1]        
        for iy in xrange(height_new):
            for ix in xrange(width_new):
                idy = iy - self._y_off
                idx = ix - self._x_off
                while idx < 0:
                    idx += self._period
                while idx >= width_old:
                    idx -= self._period
                if idy < 0 or idy >= height_old:
                    output[iy, ix] = 0.0
                else:
                    output[iy, ix] = input_[idy, idx]
        return output

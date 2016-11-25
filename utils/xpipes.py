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
import pickle
import json

import xlearn.utils.tensor as utt
import xlearn.utils.image as uti
import xlearn.utils.general as utg


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
        # for id_pipe in xrange(self.n_inputs):
        #     if self._branches[id_pipe].output_type != object:
        #         raise TypeError("Output Type mismatch of %d-th input."%id_pipe)
        pass

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
        input_ = self._gather()
        if len(self._branches) == 1:
            input_ = input_[0]
        return input_

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


class FunctionCaller(Pipe):

    def __init__(self, input_, foo, name="FunctionCaller", is_start=None, is_seal=False, *args, **kwargs):
        super(FunctionCaller, self).__init__(
            input_, name=name, is_start=is_start, is_seal=is_seal)
        self._foo = foo
        self._args = args
        self._kwargs = kwargs

    def _pump(self):
        return self._foo(*self._args, **self._kwargs)

    def _process(self):
        return self._foo(*self._args, **self._kwargs)


class SingleInput(Pipe):
    """
    Base class of pipes with only one input.
    Previous pipe must havs output, None is *NOT* allowed.
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

    def _gather_f(self):
        input_ = self._gather()
        if len(input_) != 1:
            raise ValueError(utg.errmsg(
                len(input_), 1, "Wrong input list length, "))
        output = input_[0]
        if output is None:
            raise TypeError("None input is not allowed.")
        return output


class Runner(SingleInput):
    """Drive pipes to run."""

    def __init__(self, input_, name="Runner"):
        super(Runner, self).__init__(
            input_, name=name, is_seal=False)

    def run(self):
        for _ in self._father.out:
            pass


class Counter(Pipe):
    """Counter pipe."""

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
        if len(self._branches) == 1:
            self._buffer = self._buffer[0]
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


class ListTranspose(SingleInput):

    def __init__(self, input_, name="ListTranspose", is_seal=False):
        super(ListTranspose, self).__init__(input_, name=name, is_seal=is_seal)


class Inputer(Buffer):
    """Dump given items to pipeline.
    Automtically expand list and tuple items (as a Buffer), this feature can BufferError
    shutdown by passing auto_expand=False.
    """

    def __init__(self, item=None, const_output=False, auto_expand=True, name='Inputer'):
        super(Inputer, self).__init__(
            fixed_max_state=True, name=name, is_start=True)
        self._buffer = collections.deque()
        if const_output:
            self._counter.max_state = None
        else:
            self._counter.max_state = 1
        self._auto_expand = auto_expand
        if item is not None:
            self.insert(item)

    def insert(self, item):
        if not isinstance(item, (list, tuple)) or self._auto_expand is False:
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
        if len(self._branches) == 1:
            output = self._buffer
        else:
            output = self._buffer[0]
        return output


class BatchMaker(Pipe):

    def __init__(self, input_, batch_size, pad_item=None, name="BatchMaker", is_seal=False):
        super(BatchMaker, self).__init__(
            input_, name=name, is_start=False, is_seal=is_seal)
        self._batch_size = batch_size
        self._buffer = []
        self._pad_item = pad_item

    def _dump_out(self):
        output = self._buffer[:self._batch_size]
        self._buffer = self._buffer[self._batch_size:]
        return output

    def _process(self):
        try:
            while len(self._buffer) < self._batch_size:
                input_ = self._gather()
                self._buffer.extend(input_)
        except StopIteration:
            while len(self._buffer) < self._batch_size:
                self._buffer.append(self._pad_item)
        return self._dump_out()


class RandomPrefix(Pipe):
    """Generate a valid filename with given prefix.
    """

    def __init__(self, prefix=None, name='RandomPrefix', is_seal=False):
        super(RandomPrefix, self).__init__(
            is_start=True, name=name, is_seal=is_seal)
        if prefix is None:
            self._prefix = "TMPRAND"
        else:
            self._prefix = prefix
        time_str_list = list(str(time.time()))
        time_str_list.remove('.')
        self._prefix += ''.join(time_str_list[-2:])
        self._prefix += '%08d'%random.randint(0, 99999999)
        self._counter = Counter()

    def _pump(self):
        output = self._prefix + "%09d" % (next(self._counter.out),)
        return output


class LabelFinder(SingleInput):

    def __init__(self, input_, label_foo=None, conf_file=None, name="LabelFinder", is_seal=False):
        super(LabelFinder, self).__init__(
            input_, name=name, is_seal=is_seal)
        if conf_file is None:
            self._label_foo = label_foo
            self._use_file = False
        else:
            with open(conf_file) as cfile:
                self._pair_dict = json.load(cfile)
            self._use_file = True

    def _process(self):
        data_filename = self._gather_f()
        if self._use_file:
            return self._pair_dict[data_filename]
        else:
            return self._label_foo(data_filename)


class FileNameLooper(Pipe):
    """File name iterator for a folder."""

    def __init__(self, folder,
                 prefix,
                 ids=None,
                 random_shuffle=False,
                 suffix='npy',
                 name='FileNameLooper',
                 max_epoch=None,
                 is_seal=False):
        super(FileNameLooper, self).__init__(is_start=True,
                                             name=name,
                                             is_seal=is_seal)
        self._output_type = str
        self._path = os.path.abspath(folder)

        self._is_random = random_shuffle
        self._prefix = prefix
        self._suffix = suffix
        list_all = os.listdir(self._path)
        list_all.sort()
        # file name filter
        self._file_names = []
        if ids is None:
            for file in list_all:
                prefix, id, suffix = utg.seperate_file_name(file)
                if prefix is not None:
                    self._file_names.append(file)
        else:
            for id_ in ids:
                filename = utg.form_file_name(self._prefix,
                                              id_,
                                              self._suffix)
                if filename in list_all:
                    self._file_names.append(filename)
        # counters
        self._epoch_counter = Counter(max_epoch)
        self._fid_counter = Counter(len(self._file_names))

        self._is_random = random_shuffle
        self._new_epoch()

    def _new_epoch(self):
        next(self._epoch_counter.out)
        self._fid_counter.reset()
        if self._is_random:
            random.shuffle(self._file_names)

    def _pump(self):
        filename = self._file_names[next(self._fid_counter.out)]
        fullname = os.path.join(self._path, filename)
        if self._fid_counter.is_end():
            self._new_epoch()
        return fullname

    @property
    def n_files(self):
        return self._fid_counter.max_state

    @property
    def epoch(self):
        return self._epoch_counter.state

    @property
    def max_epoch(self):
        return self._epoch_counter.max_state


class NPYReaderSingle(SingleInput):
    """A pipe warper for .npy reader."""

    def __init__(self, input_, name='NPYReader', is_seal=False):
        super(NPYReaderSingle, self).__init__(
            input_, name=name, is_seal=is_seal)

    def _process(self):
        filename = self._gather_f()
        data = np.load(filename)
        return data


class FolderReader(Pipe):
    """Load all files with valid data file name.
    Support type:
        - npy
        - dat (pickle data)
    <input> None
    <output> tensor from .npy file.
    """
    # TODO Add support for raw data.

    def __init__(self, folder,
                 prefix,
                 ids=None,
                 random_shuffle=False,
                 suffix='npy',
                 name='FileReader',
                 is_seal=False):
        super(FolderReader, self).__init__(is_start=True,
                                           name=name,
                                           is_seal=is_seal)
        self._output_type = np.ndarray
        self._path = os.path.abspath(folder)
        self._suffix = suffix
        self._filename_looper = FileNameLooper(self._path,
                                               prefix,
                                               ids=ids,
                                               random_shuffle=random_shuffle,
                                               suffix=suffix)
        self._shape = None

    def _pump(self):
        filename = next(self._filename_looper.out)
        fullname = os.path.join(self._path, filename)
        if self._suffix == 'npy':
            data = np.array(np.load(fullname))
        if self._suffix == 'dat':
            with open(fullname, 'r') as file_in:
                data = pickle.load(file_in)
        self._shape = data.shape
        return data

    @property
    def n_files(self):
        return self._filename_looper.n_files

    @property
    def last_shape(self):
        return self._shape

    @property
    def epoch(self):
        return self._filename_looper.epoch


class FolderWriter(Pipe):
    """Write a series of numpy.ndarray to npy files.
    """

    def __init__(self, path, prefix, data, count, name='NPYWriter', suffix='npy', is_seal=False):
        super(FolderWriter, self).__init__(
            [data, count], name=name, is_start=False, is_seal=is_seal)
        self._branches[1].reset()
        self._prefix = prefix
        self._path = os.path.abspath(path)
        self._suffix = suffix

    def _process(self):
        input_ = self._gather()
        data = input_[0]
        count = input_[1]
        filename = utg.form_file_name(self._prefix,
                                      count,
                                      self._suffix)
        fullname = os.path.join(self._path, filename)
        if self._suffix == 'npy':
            np.save(fullname, data)
        if self._suffix == 'dat':
            with open(fullname, 'w') as file_out:
                pickle.dump(file_out, data)


class ImageFormater(SingleInput):
    """
    Convert tensor input format which is plotable via numpy.imshow.
    """

    def __init__(self, input_, is_uint8=False, offset=3, name='ImageFormater', is_seal=False):
        super(ImageFormater, self).__init__(
            input_, name=name, is_start=False, is_seal=is_seal)
        self._is_uint8 = is_uint8
        self._offset = offset

    def _process(self):
        img_maybe = self._gather_f()
        output = uti.image_formater(
            img_maybe, is_uint8=self.is_uint8, offset=self._offset)
        return output

    @property
    def is_uint8(self):
        return self._is_uint8

    @is_uint8.setter
    def is_uint8(self, value):
        self._is_uint8 = value

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value


class ImageGrayer(SingleInput):
    """Convert rgb images into gray images.
    """

    def __init__(self, input_, name='ImageGrayer', is_seal=False):
        super(ImageGrayer, self).__init__(input_, name=name, is_seal=is_seal)

    def _process(self):
        image = self._gather_f()
        gray = uti.rgb2gray(image)
        return gray


class TensorStacker(SingleInput):

    def __init__(self, input_, name="TensorStacker", is_seal=False):
        super(TensorStacker, self).__init__(input_, name=name, is_seal=is_seal)

    def _process(self):
        tensor_list = self._gather_f()
        if not isinstance(tensor_list, (list, tuple)):
            raise TypeError('Only list of 1HW? or 1HWD? tensor is allowed.')
        oldshape = tensor_list[0].shape
        oldshape = list(oldshape)
        n_tensor = len(tensor_list)
        if oldshape[0] == 1:
            oldshape = oldshape[1:]
        newshape = [n_tensor] + oldshape
        output = np.zeros(newshape)
        dim = len(oldshape)
        for i in xrange(n_tensor):
            ind = [i]
            for j in xrange(dim):
                ind.append(slice(0, oldshape[j]))
            output[ind] = tensor_list[i]
        return output


class TensorFormater(SingleInput):
    """Reshape a tensor, e.g. image to a standard tensor format.
        new_shape: list of int, new shape, DEFAULT=None;
        squeeze: boolean, delete all dimensions with dimension size = 1, DEFAULT=False;
        auto_shape: boolean, form input into NHWC form, DEFAULT=True
    """

    def __init__(self, input_,
                 new_shape=None,
                 auto_shape=True,
                 squeeze=False,
                 name='TensorFormater', is_seal=False):
        super(TensorFormater, self).__init__(
            input_, name=name, is_seal=is_seal)
        self._shape = new_shape
        self._auto_shape = auto_shape
        self._squeeze = squeeze

    def _process(self):
        tensor = self._gather_f()
        if self._shape is not None:
            output = np.reshape(tensor, self._shape)
        elif self._auto_shape:
            output = uti.image2tensor(tensor)
        elif self._squeeze:
            preshape = tensor.shape
            newshape = list(preshape)
            while 1 in newshape:
                newshape.remove(1)
            output = np.reshape(output, newshape)
        return output


class PatchGenerator(SingleInput):

    def __init__(self, input_,
                 shape, strides, random_gen=False, n_patches=None, check_all=False,
                 name='PatchGenerator', is_seal=False):
        super(PatchGenerator, self).__init__(
            input_, name=name, is_seal=is_seal)
        self._shape = shape
        self._strides = strides
        self._random = random_gen
        self._n_patches = n_patches
        self._check_all = check_all

    def _process(self):
        input_ = self._gather_f()
        output = utt.crop_tensor(input_, self._shape,
                                 strides=self._strides,
                                 check_all=self._check_all,
                                 random_shuffle=self._random,
                                 n_patches=self._n_patches)
        return output


class ImageCropper(SingleInput):

    def __init__(self, input_, margin0, margin1, name="ImageCropper", is_seal=False):
        super(ImageCropper, self).__init__(input_, name=name, is_seal=is_seal)
        self._margin0 = margin0
        self._margin1 = margin1

    def _process(self):
        input_ = self._gather_f()
        patch_shape = list(input_.shape)
        dim = len(patch_shape)
        for i in xrange(dim):
            patch_shape[i] -= self._margin0[i]
            patch_shape[i] -= self._margin1[1]
            if patch_shape[i] < 1:
                raise ValueError(utg.errmsg(
                    patch_shape[i], ">0", "crop smaller than 0, "))
        ind = []
        for i in xrange(dim):
            ind.append(
                xrange(self._margin0[i] + 1, input_.shape[i] - self._margin1[i]))
        output = input_[ind]
        return output


class PatchMerger(SingleInput):

    def __init__(self, input_,
                 tensor_shape, patch_shape, strides,
                 name="PatchMerger", is_end=None, is_seal=False):
        super(PatchMerger, self).__init__(input_, is_seal=is_seal)
        self._tensor_shape = tensor_shape
        self._patch_shape = patch_shape
        self._strides = strides

    def _process(self):
        input_ = self._gather_f()
        output_ = utt.combine_tensor_list(
            input_, self._tensor_shape, self._strides)
        return output_


class DownSampler(SingleInput):

    def __init__(self, input_, ratio, method='mean', name='DownSampler', is_seal=False):
        super(DownSampler, self).__init__(input_, name=name, is_seal=is_seal)
        self._ratio = ratio
        self._method = method

    def _process(self):
        input_ = self._gather_f()
        output = utt.down_sample_nd(
            input_, ratio=self._ratio, method=self._method)
        return output


class TensorSlicer(SingleInput):
    """
    Slice a 4D NHW1 tensor into a list of HW tensor.
    """

    def __init__(self, input_, shape, name='TensorSlicer'):
        super(TensorSlicer, self).__init__(input_, name)
        self._shape = shape

    def _process(self):
        input_ = self._gather_f()
        output = utt.crop_tensor(input_, self._shape)
        return output


class PeriodicalPadding(SingleInput):
    """
    Padding for sinograms, input must have shape NHW1

    Zero padding for y.
    Periodical padding for x.    
    """

    def __init__(self, input_, pw_x0, pw_x1, pw_y0, pw_y1):
        super(PeriodicalPadding, self).__init__(input_)
        self._pw_x0 = pw_x0
        self._pw_x1 = pw_x1
        self._pw_y0 = pw_y0
        self._pw_y1 = pw_y1

    def _process(self):
        input_ = self._gather_f()
        pad_width = ((0, 0), (self._pw_x0, self._pw_x1), (0, 0), (0, 0))
        output = np.pad(input, pad_width=pad_width, mode='constant')
        pad_width = ((0, 0), (0, 0), (self._pw_y0, self._pw_y1), (0, 0))
        output = np.pad(input, pad_width=pad_width, mode='wrap')
        return output


class Proj2Sino(SingleInput):

    def __init__(self, input_, name='Proj2Sino', is_start=False, is_seal=False):
        super(Proj2Sino, self).__init__(
            input_, name=name, is_start=is_start, is_seal=is_seal)

    def _process(self):
        input_ = self._gather_f()
        output = uti.proj2sino(input_)
        return output


class Sino2Proj(SingleInput):

    def __init__(self, input_, name='Sino2Proj', is_start=False, is_seal=False):
        super(Sino2Proj, self).__init__(
            input_, name=name, is_start=is_start, is_seal=is_seal)

    def _process(self):
        input_ = self._gather_f()
        output = uti.sino2proj(input_)
        return output

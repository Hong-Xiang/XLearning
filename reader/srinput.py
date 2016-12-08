#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2016-11-03 10:20:52

@author: HongXiang

General image inputer
"""
from __future__ import absolute_import, division, print_function
import os
import itertools
import json
import numpy as np

from six.moves import xrange
import xlearn.utils.xpipes as utp
import xlearn.utils.tensor as utt
import xlearn.utils.general as utg
import xlearn.utils.image as uti
import itertools


def label_name(data_name, case_digit=None, label_prefix=None):
    prefix, id, suffix = utg.seperate_file_name(data_name)
    if case_digit is not None:
        id = str(id)
        id = id[:-case_digit]
        id = int(id)
    if label_prefix is None:
        label_prefix = 'label'
    output = utg.form_file_name(label_prefix, id, suffix)
    return output


def config_file_generator(conf_file,
                          path_data, prefix_data,
                          path_label, prefix_label,
                          patch_shape,
                          strides,
                          batch_size,
                          same_file_data_label=False,
                          single_file=False,
                          filename=None,
                          n_patch_per_file=None,
                          down_sample_ratio=1,
                          ids=None,
                          padding=False,
                          dataset_type='test',
                          down_sample_method='fixed',
                          random_shuffle=False,
                          mean=1,
                          std=128,
                          **kwargs):
    """generate config .JSON file."""
    data = dict()
    data.update({'path_data': path_data})
    data.update({'prefix_data': prefix_data})
    data.update({'path_label': path_label})
    data.update({'prefix_label': prefix_label})
    data.update({'patch_shape': patch_shape})
    data.update({'strides': strides})
    data.update({'batch_size': batch_size})
    data.update({'same_file_data_label': same_file_data_label})
    data.update({'single_file': single_file})
    data.update({'filename': filename})
    data.update({'n_patches': n_patch_per_file})
    data.update({'down_sample_ratio': down_sample_ratio})
    data.update({'ids': ids})
    data.update({'padding': padding})
    data.update({'dataset_type': dataset_type})
    data.update({'down_sample_method': down_sample_method})
    data.update({'random_shuffle': random_shuffle})
    data.update({'need_gray': True})
    data.update({'check_all': False})
    data.update({'mean': mean})
    data.update({'std': std})
    data.update(kwargs)
    with open(conf_file, 'w') as fp:
        json.dump(data, fp, sort_keys=True, indent=4, separators=(',', ': '))


class DataSetSR(object):
    """
    A general super resolution net.
    Args:
    - config_file: a .JSON file containing configurations.

    next_batch() provides
            (low_resolution_patch_tensor, high_resolution_patch_tensor)
    """

    def __init__(self, **kwargs):
        if "conf" in kwargs:
            with open(kwargs["conf"]) as conf_file:
                paras = json.load(conf_file)
        else:
            paras = kwargs

        self._patch_shape = paras['patch_shape']
        self._batch_size = paras['batch_size']
        self._dataset_type = paras['dataset_type']
        self._ratio = paras['down_sample_ratio']
        self._padding = paras['padding']
        self._dataset_type = paras['dataset_type']
        self._strides = paras['strides']
        self._std = paras['std']
        self._mean = paras['mean']
        self._eps = paras['eps']
        self._data_filename_iter = utp.FileNameLooper(paras['path_data'],
                                                      prefix=paras[
                                                          'prefix_data'],
                                                      random_shuffle=paras[
                                                          'random_shuffle'],
                                                      ids=paras['ids'])
        self._nzratio = paras['nzratio']
        if paras['same_file_data_label']:
            self._data_image = utp.NPYReaderSingle(self._data_filename_iter)
            self._data_image_copyer = utp.Copyer(
                self._data_image, copy_number=2)
            self._data_tensor = utp.TensorFormater(self._data_image_copyer)
            self._label_tensor = utp.TensorFormater(self._data_image_copyer)
        else:
            self._data_filename_copyer = utp.Copyer(
                self._data_filename_iter, copy_number=2)
            if paras['same_file_data_label']:
                self._label_filename = utp.Pipe(self._data_filename_copyer)
            else:
                self._label_filename = utp.LabelFinder(
                    self._data_filename_copyer, label_name)
            self._data_filename = utp.Pipe(self._data_filename_copyer)
            self._data_image = utp.NPYReaderSingle(self._data_filename)
            self._label_image = utp.NPYReaderSingle(self._label_filename)
            self._data_tensor = utp.TensorFormater(self._data_image)
            self._label_tensor = utp.TensorFormater(self._label_image)
        if paras['need_gray']:
            self._data_gray = utp.ImageGrayer(self._data_tensor)
            self._label_gray = utp.ImageGrayer(self._label_tensor)
            self._merge = utp.Pipe([self._data_gray, self._label_gray])
            self._stacker = utp.TensorStacker(self._merge)
        else:
            self._merge = utp.Pipe([self._data_tensor, self._label_tensor])
            self._stacker = utp.TensorStacker(self._merge)
        self._tensor = utp.TensorFormater(self._stacker)
        patch_shape = paras['patch_shape']
        patch_shape_2 = patch_shape[:]
        patch_shape_2[0] = 2

        self._patch_generator = utp.PatchGenerator(self._tensor,
                                                   shape=patch_shape_2,
                                                   n_patches=paras[
                                                       'n_patches'],
                                                   strides=paras[
                                                       'strides'],
                                                   random_gen=paras[
                                                       'random_shuffle'],
                                                   check_all=paras['check_all'])

        self._buffer = utp.Buffer(self._patch_generator)

        self._slicer = utp.TensorSlicer(self._buffer, patch_shape)

        self._buffer2 = utp.Buffer(self._slicer)

        self._hr_patch_gen = utp.TensorFormater(self._buffer2)

        self._down_sample = utp.DownSampler(
            self._buffer2, self._ratio, method=paras['down_sample_method'])

        self._lr_patch_gen = utp.TensorFormater(self._down_sample)

    def next_batch(self):
        """Generate next batch data, padding zeros, and whiten."""
        high_shape = [self._batch_size, self._patch_shape[
            1], self._patch_shape[2], self._patch_shape[3]]
        if self._padding:
            low_height = int(np.ceil(self._patch_shape[1] / self._ratio[1]))
            low_width = int(np.ceil(self._patch_shape[2] / self._ratio[2]))
        else:
            low_height = int(self._patch_shape[1] / self._ratio[1])
            low_width = int(self._patch_shape[2] / self._ratio[2])
        low_shape = [self._batch_size, low_height,
                     low_width, self._patch_shape[3]]
        high_tensor = np.zeros(high_shape)
        low_tensor = np.zeros(low_shape)
        cid = 0
        for i in xrange(self._batch_size):
            # for patch_high, patch_low in
            # itertools.izip(self._hr_patch_gen.out, self._lr_patch_gen.out):
            try:
                while True:
                    patch_high = next(self._hr_patch_gen.out)
                    patch_low = next(self._lr_patch_gen.out)
                    total_pixel = np.size(patch_high)
                    total_nonez = np.count_nonzero(patch_high)
                    nnz_ratio = np.float(total_nonez) / np.float(total_pixel)
                    if nnz_ratio >= self._nzratio:
                        break
            except StopIteration:
                break
            pstd = np.std(patch_high)
            pstd = np.max([pstd, 1.0])
            patch_high /= pstd
            patch_low /= pstd
            pmean = np.mean(patch_high)
            patch_high -= pmean
            patch_low -= pmean
            high_tensor[cid, :, :, :] = patch_high
            low_tensor[cid, :, :, :] = patch_low
            cid += 1
            # if cid == self._batch_size:
            #     break

        # low_tensor /= self._std
        # low_tensor -= self._mean
        # high_tensor /= self._std
        # high_tensor -= self._mean

        return low_tensor, high_tensor

    def form_image(self, patches, image_shape, strides=None):
        if strides is None:
            strides = self._strides
        output = utt.combine_tensor_list(patches, image_shape, strides)
        return output

    @property
    def height(self):
        return self._patch_shape[0]

    @property
    def width(self):
        return self._patch_shape[1]

    @property
    def strides(self):
        return self._strides

    @property
    def epoch(self):
        """current epoch"""
        return self._data_filename_iter.epoch


class DataSetSRInfer(object):

    def __init__(self, config_file, filename=None):
        with open(config_file) as conf_file:
            paras = json.load(conf_file)
        self._path_input = os.path.abspath(paras['path_input'])
        if filename is None:
            self._filename_input = paras['filename_input']
        else:
            self._filename_input = filename
        self._need_gray = paras['need_gray']

        self._batch_size = paras['batch_size']
        self._patch_shape = paras['patch_shape']
        self._strides = paras['strides']
        self._ratio = paras['down_sample_ratio']
        self._method = paras['down_sample_method']

        self._std = paras['std']
        self._mean = paras['mean']

        self._path_output = os.path.abspath(paras['path_output'])
        self._filename_output = paras['filename_output']

        self._patch_shape_low = list(
            map(lambda x: x[0] // x[1], zip(self._patch_shape, self._ratio)))

        self._image = None
        self._recon = None

        fullname = os.path.join(self._path_input, self._filename_input)
        self.load_new_file(fullname)

    def load_new_file(self, fullname):
        self._filename = fullname
        self._image = np.array(np.load(self._filename))
        self._image = uti.image2tensor(self._image)
        if self._need_gray:
            self._image = uti.rgb2gray(self._image)
        self._patches = utt.crop_tensor(
            self._image, self._patch_shape, self._strides, check_all=False)
        self._down_patches = []
        for patch in self._patches:
            self._down_patches.append(utt.down_sample_nd(patch, self._ratio))
        self._n_patch = len(self._down_patches)
        self._n_batch = int(
            np.ceil(np.float(self._n_patch) / np.float(self._batch_size)))
        self._stds = []
        self._means = []
        for i in xrange(self._n_patch):
            pstd = np.std(self._down_patches[i])
            pstd = np.max([pstd, 1.0])
            self._down_patches[i] /= pstd
            self._stds.append(pstd)
            pmean = np.mean(self._down_patches[i])
            self._down_patches[i] -= pmean
            self._means.append(pmean)
        self._cid = 0
        self._result = []

    def next_batch(self):
        """Generate next batch data, padding zeros, and whiten."""
        low_shape = [self._batch_size, self._patch_shape_low[1],
                     self._patch_shape_low[2], self._patch_shape_low[3]]
        low_tensor = np.zeros(low_shape)
        for i in xrange(self._batch_size):
            if self._cid < self._n_patch:
                low_tensor[i, :, :, :] = self._down_patches[self._cid]
                self._cid += 1
        return low_tensor

    def add_result(self, tensor):
        n_patches = tensor.shape[0]
        sr_patch_shape = list(tensor.shape)
        sr_patch_shape[0] = 1
        for i in xrange(n_patches):
            tmppatch = tensor[i, :, :, :]
            tmppatch = np.reshape(tmppatch, sr_patch_shape)
            self._result.append(tmppatch)

    def form_image(self):
        """Reconstruct pathches to image.
        If strides is None, then non crop net is assumed.
        """
        patches = self._result[:self._n_patch]
        sr_patch_shape = patches[0].shape
        margin0 = list(
            map(lambda x: (x[0] - x[1]) / 2, zip(self._patch_shape, sr_patch_shape)))
        
        margin1 = list(
            map(lambda x: (x[0] - x[1]) / 2, zip(self._patch_shape, sr_patch_shape)))
        margin1_last = list(map(lambda x: (
            x[1] - x[0] % x[1]) % x[1], zip(self._image.shape, self._patch_shape)))
        margin1 = list(map(lambda x: x[0] + x[1], zip(margin1, margin1_last)))

        margin0 = list(map(int, margin0))        
        margin1 = list(map(int, margin1))
        # margin0 = [0, 0, 0, 0]
        # margin1 = [0, 0, 51, 0]        
        patches = self._result
        for i in xrange(self._n_patch):
            patches[i] += self._means[i]
            patches[i] *= self._stds[i]
        output = utt.combine_tensor_list(
            patches, shape=self._image.shape, strides=self._strides, margin0=margin0, margin1=margin1)
        self._recon = output
        return output

    def save_result(self):
        fullname = os.path.join(self._path_output, self._filename_output)
        np.save(fullname, self._recon)

    @property
    def height(self):
        return self._patch_shape[0]

    @property
    def width(self):
        return self._patch_shape[1]

    @property
    def strides(self):
        return self._strides

    @property
    def n_batch(self):
        return self._n_batch

    @property
    def image(self):
        return self._image

    @property
    def recon(self):
        return self._recon

    @property
    def path_infer(self):
        return self._path_input

    @property
    def path_output(self):
        return self._path_output

#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 2016-11-03 10:20:52

@author: HongXiang

General image inputer
"""
import os
import json
import numpy as np
import logging
import xlearn.utils.general as utg
import xlearn.utils.tensor as utt
import xlearn.utils.image as uti
import xlearn.utils.xpipes as utp

import xlearn.reader.base


class DataSetSuperResolution(xlearn.reader.base.DataSet):
    """DataSet for image super resolution
    Generate downsampled low resolution image as data
    Generate high resolution image as label

    Multiple input mode is supported:
    1.  is_single:
        True: only one image is provided, dataset will perform downsample;
        False: two files are provided
    2.  is_patch:
        True: a full image is provided, dataset will perform patch generation;
        False: patches are provided, no patch generation.

    Reconstruction routine:
    Requires:
        is_patch == False
        is_recon == True
    Dataset will generate pathches from one image, wait for respond tensor,
    and try to combine them into a large image.

    Args [DEFAULT]:
    -   is_single ['True']: dataset mode
    -   path_data: path of data
    -   prefix_data ['img']: prefix of data
    """

    # TODO: Implement two file mode
    # TODO: Implement patch mode

    def __init__(self, filenames=None, **kwargs):
        super(DataSetSuperResolution, self).__init__(filenames, **kwargs)

    def _get_default_paras(self):
        default_paras = super(DataSetSuperResolution,
                              self)._get_default_paras()
        default_paras.update({
            'prefix_data': 'img',
            'is_single': True,
            'is_patch': False,
            'is_recon': False,
            'is_multi_channel': False,
            'is_lock': False
        })
        return default_paras

    def _gather_paras_common(self):
        super(DataSetSuperResolution, self)._gather_paras_common()

        # file settings:
        self._is_single = self._paras['is_single']
        self._is_recon = self._paras['is_recon']
        self._path_data = self._paras['path_data']
        self._prefix_data = self._paras['prefix_data']
        if not self._is_single:
            self._path_label = self._paras['path_label']
            self._prefix_lable = self._paras['prefix_label']

        if 'ids' in self._paras:
            self._ids = self._paras['ids']
        else:
            self._ids = None

        self._is_multi_channel = self._paras['is_multi_channel']
        self._is_need_gray = self._paras['is_need_gray']

        # patch generation paras:
        self._n_patches = None
        if 'n_patches' in self._paras and self._dataset_type != "infer":
            self._n_patches = self._paras['n_patches']
        self._is_patch = self._paras['is_patch']
        self._is_check_all = self._paras['is_check_all']
        self._is_lock = self._paras['is_lock']

        # down sample paras:
        self._patch_shape = [1] + list(self._paras['shape_i'])
        self._down_sample_ratio = [1] + list(self._paras['down_sample_ratio'])
        self._down_sample_method = self._paras['down_sample_method']
        self._strides = [1] + list(self._paras['strides'])
        self._down_sample_is_pad = self._paras['down_sample_is_pad']

        # sample threshold paras:
        self._nzratio = self._paras['nzratio']
        self._eps = self._paras['eps']

        # post processing paras:
        self._std = self._paras['std']
        self._mean = self._paras['mean']
        self._is_norm = self._paras['is_norm']
        self._is_gamma = self._paras['is_gamma']
        self._gamma_r = self._paras['gamma_r']

    def _prepare(self):
        super(DataSetSuperResolution, self)._prepare()
        if not self._is_lock:
            data_filename_iter = utp.FileNameLooper(self._path_data,
                                                    prefix=self._prefix_data,
                                                    random_shuffle=self._is_shuffle,
                                                    ids=self._ids,
                                                    max_epoch=self._epoch_max)
            self._filename_iter = data_filename_iter
        else:
            self._filename_iter = utp.FileNameLooper(self._path_data,
                                                     prefix=self._prefix_data,
                                                     random_shuffle=False,
                                                     ids=self._ids,
                                                     max_epoch=1)
            data_filename_iter = utp.Inputer()
            self._filename_inputer = data_filename_iter
            self._is_next_file = True
        if self._is_single:
            data_image = utp.NPYReaderSingle(data_filename_iter)
            data_image_copyer = utp.Copyer(data_image, copy_number=2)
            data_tensor = utp.TensorFormater(data_image_copyer)
            label_tensor = utp.TensorFormater(data_image_copyer)
        else:
            data_filename_copyer = utp.Copyer(
                data_filename_iter, copy_number=2)
            label_filename = utp.LabelFinder(
                data_filename_copyer, utg.label_name)
            data_filename = utp.Pipe(data_filename_copyer)
            data_image = utp.NPYReaderSingle(data_filename)
            label_image = utp.NPYReaderSingle(label_filename)
            data_tensor = utp.TensorFormater(data_image)
            label_tensor = utp.TensorFormater(label_image)
        if self._is_need_gray:
            data_full = utp.ImageGrayer(data_tensor)
            label_full = utp.ImageGrayer(label_tensor)
        else:
            data_full = data_tensor
            label_full = label_tensor

        merge = utp.Pipe([data_full, label_full])
        stacker = utp.TensorStacker(merge)
        multi_tensor = utp.TensorFormater(stacker)
        stacked_shape = [2] + list(self._shape_o)
        patch_generator = utp.PatchGenerator(multi_tensor,
                                             shape=stacked_shape,
                                             n_patches=self._n_patches,
                                             strides=self._strides,
                                             random_gen=self._is_shuffle,
                                             check_all=self._is_check_all)
        buffer_stacked = utp.Buffer(patch_generator)

        slicer = utp.TensorSlicer(buffer_stacked, self._shape_sample_o)
        buffer_hl = utp.Buffer(slicer)
        self._label = utp.TensorFormater(buffer_hl)
        down_sample = utp.DownSampler(
            buffer_hl, self._down_sample_ratio, method=self._down_sample_method)
        self._data = utp.TensorFormater(down_sample)
        self._testo = self._filename_iter

    def test_debug(self):
        return next(self._testo.out)
        pass

    def _sample(self):
        try:
            data = next(self._data.out)
            label = next(self._label.out)
            data = np.reshape(data, self._shape_sample_i)
            label = np.reshape(label, self._shape_sample_o)
            if self._is_norm:
                # stdv = np.std(data)
                # if stdv > self._eps:
                #     data /= stdv
                #     label /= stdv
                # meanv = np.mean(data)
                # data -= meanv
                # label -= meanv
                data /= self._std
                label /= self._std
                data -= self._mean
                label -= self._mean
                if self._is_gamma:
                    data = np.power(data, self._gamma_r)
                    label = np.power(label, self._gamma_r)
        except StopIteration:
            if self._is_lock:
                raise xlearn.reader.base.EndSingleFile
            else:
                raise xlearn.reader.base.EndEpoch
            cepoch = self._filename_iter.epoch
            while cepoch > self.epoch:
                self._next_epoch()
        return data, label

    def _next_file(self):        
        try:
            super(DataSetSuperResolution, self)._next_file()
            filename = next(self._filename_iter.out)
        except StopIteration:            
            raise xlearn.reader.base.EndEpoch
        self._image = np.array(np.load(filename))
        logging.getLogger(__name__).debug("processing: " + filename)
        self._filename_inputer.insert(filename)
        self._is_next_file = False

    def free_lock(self):
        self._is_next_file = True

    @property
    def height_low(self):
        return self._shape_i[0]

    @property
    def width_low(self):
        return self._shape_i[1]

    @property
    def height_high(self):
        return self._shape_o[0]

    @property
    def width_high(self):
        return self._shape_o[1]

    @property
    def strides(self):
        return self._strides

    @property
    def is_next_file(self):
        return self._is_next_file

    @property
    def n_files(self):
        return self._filename_iter.n_files

    @property
    def is_lock(self):
        return self._is_lock

    @property
    def last_file(self):
        return self._filename_iter.last_name


class ImageReconstructer(object):

    def __init__(self, filenames=None, **kwargs):
        self._paras = utg.merge_settings(filenames=filenames, **kwargs)
        self._buffer = []
        self._margin0 = self._paras['infer_margin0']
        self._margin1 = self._paras['infer_margin1']
        self._offset0 = self._paras['infer_offset0']
        self._offset1 = self._paras['infer_offset1']
        self._is_norm = self._paras['is_norm']
        self._is_gamma = self._paras['is_gamma']
        self._gamma_r = self._paras['gamma_r']
        self._mean = self._paras['mean']
        self._std = self._paras['std']
        self._shape = self._paras['infer_shape']
        self._strides = self._paras['strides']
        if len(self._strides) < len(self._margin0):
            self._strides = [1] + list(self._strides)
        self._recon = None
        self._path_output = self._paras['path_output']
        logging.getLogger(__name__).debug(
            "ImageReconstructer end of __init__, para_string():")
        logging.getLogger(__name__).debug(self._para_string())

    def add_result(self, input_):
        """Add infer result to buffer
        """
        n_batch = input_.shape[0]        
        for i in range(n_batch):
            tmp = input_[i, :]
            tmp = np.reshape(
                tmp, [1, input_.shape[1], input_.shape[2], input_.shape[3]])
            if self._is_norm:
                if self._is_gamma:
                    tmp = np.power(tmp, -self._gamma_r)
                tmp += self._mean
                tmp *= self._std
            self._buffer.append(tmp)

    def shape_set(self, shape):
        self._shape = shape

    def reconstruction(self):
        """Reconstruct image from infered patches.
        """
        patches = self._buffer
        output = utt.combine_tensor_list(patches, shape=self._shape,
                                         strides=self._strides,
                                         margin0=self._margin0,
                                         margin1=self._margin1,
                                         check_all=True)
        self._recon = output
        self._buffer = []
        return output

    def _para_string(self):
        dic_sorted = sorted(self._paras.items(), key=lambda t: t[0])
        fmt = r"{0}: {1},"
        msg = 'DataSet Settings:\n' + \
            '\n'.join([fmt.format(item[0], item[1]) for item in dic_sorted])
        return msg

    @property
    def path_output(self):
        return self._path_output
# class DataSetSR(object):
#     # raise DeprecationWarning
#     """
#     A general super resolution net.
#     Args:
#     - config_file: a .JSON file containing configurations.

#     next_batch() provides
#             (low_resolution_patch_tensor, high_resolution_patch_tensor)
#     """

#     def __init__(self, **kwargs):
#         if "conf" in kwargs:
#             with open(kwargs["conf"]) as conf_file:
#                 paras = json.load(conf_file)
#         else:
#             paras = kwargs

#         self._patch_shape = paras['patch_shape']
#         self._batch_size = paras['batch_size']
#         self._dataset_type = paras['dataset_type']
#         self._ratio = paras['down_sample_ratio']
#         self._padding = paras['padding']
#         self._dataset_type = paras['dataset_type']
#         self._strides = paras['strides']
#         self._std = paras['std']
#         self._mean = paras['mean']
#         self._eps = paras['eps']
#         self._data_filename_iter = utp.FileNameLooper(paras['path_data'],
#                                                       prefix=paras[
#                                                           'prefix_data'],
#                                                       random_shuffle=paras[
#                                                           'random_shuffle'],
#                                                       ids=paras['ids'])
#         self._nzratio = paras['nzratio']
#         if paras['same_file_data_label']:
#             self._data_image = utp.NPYReaderSingle(self._data_filename_iter)
#             self._data_image_copyer = utp.Copyer(
#                 self._data_image, copy_number=2)
#             self._data_tensor = utp.TensorFormater(self._data_image_copyer)
#             self._label_tensor = utp.TensorFormater(self._data_image_copyer)
#         else:
#             self._data_filename_copyer = utp.Copyer(
#                 self._data_filename_iter, copy_number=2)
#             if paras['same_file_data_label']:
#                 self._label_filename = utp.Pipe(self._data_filename_copyer)
#             else:
#                 self._label_filename = utp.LabelFinder(
#                     self._data_filename_copyer, label_name)
#             self._data_filename = utp.Pipe(self._data_filename_copyer)
#             self._data_image = utp.NPYReaderSingle(self._data_filename)
#             self._label_image = utp.NPYReaderSingle(self._label_filename)
#             self._data_tensor = utp.TensorFormater(self._data_image)
#             self._label_tensor = utp.TensorFormater(self._label_image)
#         if paras['need_gray']:
#             self._data_gray = utp.ImageGrayer(self._data_tensor)
#             self._label_gray = utp.ImageGrayer(self._label_tensor)
#             self._merge = utp.Pipe([self._data_gray, self._label_gray])
#             self._stacker = utp.TensorStacker(self._merge)
#         else:
#             self._merge = utp.Pipe([self._data_tensor, self._label_tensor])
#             self._stacker = utp.TensorStacker(self._merge)
#         self._tensor = utp.TensorFormater(self._stacker)
#         patch_shape = paras['patch_shape']
#         patch_shape_2 = patch_shape[:]
#         patch_shape_2[0] = 2

#         self._patch_generator = utp.PatchGenerator(self._tensor,
#                                                    shape=patch_shape_2,
#                                                    n_patches=paras[
#                                                        'n_patches'],
#                                                    strides=paras[
#                                                        'strides'],
#                                                    random_gen=paras[
#                                                        'random_shuffle'],
#                                                    check_all=paras['check_all'])

#         self._buffer = utp.Buffer(self._patch_generator)

#         self._slicer = utp.TensorSlicer(self._buffer, patch_shape)

#         self._buffer2 = utp.Buffer(self._slicer)

#         self._hr_patch_gen = utp.TensorFormater(self._buffer2)

#         self._down_sample = utp.DownSampler(
#             self._buffer2, self._ratio, method=paras['down_sample_method'])

#         self._lr_patch_gen = utp.TensorFormater(self._down_sample)

#     def next_batch(self):
#         """Generate next batch data, padding zeros, and whiten."""
#         high_shape = [self._batch_size, self._patch_shape[
#             1], self._patch_shape[2], self._patch_shape[3]]
#         if self._padding:
#             low_height = int(np.ceil(self._patch_shape[1] / self._ratio[1]))
#             low_width = int(np.ceil(self._patch_shape[2] / self._ratio[2]))
#         else:
#             low_height = int(self._patch_shape[1] / self._ratio[1])
#             low_width = int(self._patch_shape[2] / self._ratio[2])
#         low_shape = [self._batch_size, low_height,
#                      low_width, self._patch_shape[3]]
#         high_tensor = np.zeros(high_shape)
#         low_tensor = np.zeros(low_shape)
#         cid = 0
#         for i in xrange(self._batch_size):
#             # for patch_high, patch_low in
#             # itertools.izip(self._hr_patch_gen.out, self._lr_patch_gen.out):
#             try:
#                 while True:
#                     patch_high = next(self._hr_patch_gen.out)
#                     patch_low = next(self._lr_patch_gen.out)
#                     total_pixel = np.size(patch_high)
#                     total_nonez = np.count_nonzero(patch_high)
#                     nnz_ratio = np.float(total_nonez) / np.float(total_pixel)
#                     if nnz_ratio >= self._nzratio:
#                         break
#             except StopIteration:
#                 break
#             pstd = np.std(patch_high)
#             pstd = np.max([pstd, 1.0])
#             patch_high /= pstd
#             patch_low /= pstd
#             pmean = np.mean(patch_high)
#             patch_high -= pmean
#             patch_low -= pmean
#             high_tensor[cid, :, :, :] = patch_high
#             low_tensor[cid, :, :, :] = patch_low
#             cid += 1
#             # if cid == self._batch_size:
#             #     break

#         # low_tensor /= self._std
#         # low_tensor -= self._mean
#         # high_tensor /= self._std
#         # high_tensor -= self._mean

#         return low_tensor, high_tensor

#     def form_image(self, patches, image_shape, strides=None):
#         if strides is None:
#             strides = self._strides
#         output = utt.combine_tensor_list(patches, image_shape, strides)
#         return output

#     @property
#     def height(self):
#         return self._patch_shape[0]

#     @property
#     def width(self):
#         return self._patch_shape[1]

#     @property
#     def strides(self):
#         return self._strides

#     @property
#     def epoch(self):
#         """current epoch"""
#         return self._data_filename_iter.epoch


# class DataSetSRInfer(object):

#     def __init__(self, config_file, filename=None):
#         with open(config_file) as conf_file:
#             paras = json.load(conf_file)
#         self._path_input = os.path.abspath(paras['path_input'])
#         if filename is None:
#             self._filename_input = paras['filename_input']
#         else:
#             self._filename_input = filename
#         self._need_gray = paras['need_gray']

#         self._batch_size = paras['batch_size']
#         self._patch_shape = paras['patch_shape']
#         self._strides = paras['strides']
#         self._ratio = paras['down_sample_ratio']
#         self._method = paras['down_sample_method']

#         self._std = paras['std']
#         self._mean = paras['mean']

#         self._path_output = os.path.abspath(paras['path_output'])
#         self._filename_output = paras['filename_output']

#         self._patch_shape_low = list(
#             map(lambda x: x[0] // x[1], zip(self._patch_shape, self._ratio)))

#         self._image = None
#         self._recon = None

#         fullname = os.path.join(self._path_input, self._filename_input)
#         self.load_new_file(fullname)

#     def load_new_file(self, fullname):
#         self._filename = fullname
#         self._image = np.array(np.load(self._filename))
#         self._image = uti.image2tensor(self._image)
#         if self._need_gray:
#             self._image = uti.rgb2gray(self._image)
#         self._patches = utt.crop_tensor(
#             self._image, self._patch_shape, self._strides, check_all=False)
#         self._down_patches = []
#         for patch in self._patches:
#             self._down_patches.append(utt.down_sample_nd(patch, self._ratio))
#         self._n_patch = len(self._down_patches)
#         self._n_batch = int(
#             np.ceil(np.float(self._n_patch) / np.float(self._batch_size)))
#         self._stds = []
#         self._means = []
#         for i in xrange(self._n_patch):
#             pstd = np.std(self._down_patches[i])
#             pstd = np.max([pstd, 1.0])
#             self._down_patches[i] /= pstd
#             self._stds.append(pstd)
#             pmean = np.mean(self._down_patches[i])
#             self._down_patches[i] -= pmean
#             self._means.append(pmean)
#         self._cid = 0
#         self._result = []

#     def next_batch(self):
#         """Generate next batch data, padding zeros, and whiten."""
#         low_shape = [self._batch_size, self._patch_shape_low[1],
#                      self._patch_shape_low[2], self._patch_shape_low[3]]
#         low_tensor = np.zeros(low_shape)
#         for i in xrange(self._batch_size):
#             if self._cid < self._n_patch:
#                 low_tensor[i, :, :, :] = self._down_patches[self._cid]
#                 self._cid += 1
#         return low_tensor

#     def add_result(self, tensor):
#         n_patches = tensor.shape[0]
#         sr_patch_shape = list(tensor.shape)
#         sr_patch_shape[0] = 1
#         for i in xrange(n_patches):
#             tmppatch = tensor[i, :, :, :]
#             tmppatch = np.reshape(tmppatch, sr_patch_shape)
#             self._result.append(tmppatch)

#     def form_image(self):
#         """Reconstruct pathches to image.
#         If strides is None, then non crop net is assumed.
#         """
#         patches = self._result[:self._n_patch]
#         sr_patch_shape = patches[0].shape
#         margin0 = list(
# map(lambda x: (x[0] - x[1]) / 2, zip(self._patch_shape,
# sr_patch_shape)))

#         margin1 = list(
#             map(lambda x: (x[0] - x[1]) / 2, zip(self._patch_shape, sr_patch_shape)))
#         margin1_last = list(map(lambda x: (
#             x[1] - x[0] % x[1]) % x[1], zip(self._image.shape, self._patch_shape)))
#         margin1 = list(map(lambda x: x[0] + x[1], zip(margin1, margin1_last)))

#         margin0 = list(map(int, margin0))
#         margin1 = list(map(int, margin1))
#         # margin0 = [0, 0, 0, 0]
#         # margin1 = [0, 0, 51, 0]
#         patches = self._result
#         for i in xrange(self._n_patch):
#             patches[i] += self._means[i]
#             patches[i] *= self._stds[i]
#         output = utt.combine_tensor_list(
#             patches, shape=self._image.shape, strides=self._strides, margin0=margin0, margin1=margin1)
#         self._recon = output
#         return output

#     def save_result(self):
#         fullname = os.path.join(self._path_output, self._filename_output)
#         np.save(fullname, self._recon)

#     @property
#     def height(self):
#         return self._patch_shape[0]

#     @property
#     def width(self):
#         return self._patch_shape[1]

#     @property
#     def strides(self):
#         return self._strides

#     @property
#     def n_batch(self):
#         return self._n_batch

#     @property
#     def image(self):
#         return self._image

#     @property
#     def recon(self):
#         return self._recon

#     @property
#     def path_infer(self):
#         return self._path_input

#     @property
#     def path_output(self):
#         return self._path_output

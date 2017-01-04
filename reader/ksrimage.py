import os
import json
import numpy as np
import logging
import xlearn.utils.general as utg
import xlearn.utils.tensor as utt
import xlearn.utils.image as uti
import xlearn.utils.xpipes as utp
import xlearn.reader.base


class DataGenerator(object):
    """ image generator for keras
    """

    def __init__(self, filenames=None, **kwargs):
        self._settings = utg.merge_settings(filenames=filenams, **kwargs)

        self._path_data = self._paras['path_data']
        self._prefix_data = self._paras['prefix_data']

        self._is_need_gray = self._paras['is_need_gray']

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
        self._norm_method = self._paras['norm_method']
        self._nzratio = self._paras['nzratio']

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
        self._means = []
        self._stds = []

    def test_debug(self):
        return next(self._testo.out)
        pass

    def _batch_start(self):
        super(DataSetSuperResolution, self)._batch_start()
        self._means = []
        self._stds = []

    def _sample(self):
        try:
            while True:
                data = next(self._data.out)
                label = next(self._label.out)
                total_pixel = np.size(data)
                total_nonez = np.count_nonzero(data)
                nnz_ratio = np.float(total_nonez) / np.float(total_pixel)
                if nnz_ratio >= self._nzratio:
                    break
            data = np.reshape(data, self._shape_sample_i)
            label = np.reshape(label, self._shape_sample_o)
            if self._is_norm:
                if self._norm_method == "global":
                    data /= self._std
                    label /= self._std
                    data -= self._mean
                    label -= self._mean
                elif self._norm_method == "patch":
                    stdv = np.std(data)
                    if stdv > self._eps:
                        data /= stdv
                        label /= stdv
                    else:
                        stdv = 1.0
                    meanv = np.mean(data)
                    data -= meanv
                    label -= meanv
                    self._means.append(meanv)
                    self._stds.append(stdv)
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
        self._is_no_more_sample = False

    def moments(self):
        while len(self._means) < self.batch_size:
            self._means.append(0.0)
            self._stds.append(1.0)
        return self._means, self._stds

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

    @property
    def norm_method(self):
        return self._norm_method

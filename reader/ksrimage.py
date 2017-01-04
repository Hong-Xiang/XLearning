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
        self._settings = utg.merge_settings(filenames=filenames, **kwargs)

        self._path_data = self._settings['path_data']
        self._prefix_data = self._settings['prefix_data']
        self._is_need_gray = self._settings.get('is_need_gray', False)
        self._is_shuffle = self._settings.get('is_shuffle', True)
        self._shape_i = self._settings['shape_i']
        self._shape_o = self._settings['shape_o']

        # down sample paras:
        self._down_sample_ratio = [1] + \
            list(self._settings['down_sample_ratio'])
        self._down_sample_method = self._settings['down_sample_method']
        self._strides = [1] + list(self._settings['strides'])
        self._down_sample_is_pad = self._settings['down_sample_is_pad']

        # sample threshold paras:
        self._nzratio = self._settings.get('nzratio', 0.0)
        self._eps = self._settings.get('eps', 1e-5)

        # post processing paras:
        self._std = self._settings['std']
        self._mean = self._settings['mean']
        self._is_norm = self._settings.get('is_norm', True)
        self._is_norm_gamma = self._settings.get('is_gamma', False)
        self._norm_gamma_r = self._settings.get('gamma_r', 0.3)

        data_filename_iter = utp.FileNameLooper(self._path_data,
                                                prefix=self._prefix_data,
                                                random_shuffle=self._is_shuffle,
                                                max_epoch=-1)
        self._filename_iter = data_filename_iter

        data_image = utp.NPYReaderSingle(data_filename_iter)
        data_image_copyer = utp.Copyer(data_image, copy_number=2)
        data_tensor = utp.TensorFormater(data_image_copyer)
        label_tensor = utp.TensorFormater(data_image_copyer)
        if self._is_need_gray:
            data_full = utp.ImageGrayer(data_tensor)
            label_full = utp.ImageGrayer(label_tensor)
        else:
            data_full = data_tensor
            label_full = label_tensor

        down_sample = utp.DownSampler(
            data_full, self._down_sample_ratio, method=self._down_sample_method)

        self._data = utp.TensorFormater(down_sample)
        self._label = utp.TensorFormater(label_full)

    def __next__(self):
        data, label = next(zip(self._data.out, self._label.out))
        if self._is_norm:
            std = np.max(data) / 2.0
            data /= std
            label /= std
            data -= 1
            label -= 1
            if self._is_norm_gamma:
                # TODO: Implementation
                pass
        data = data[0, :self._shape_i[0], :self._shape_i[1], 0]
        label = label[0, :self._shape_o[0], :self._shape_o[1], 0]
        return data, label

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
    def n_files(self):
        return self._filename_iter.n_files

    @property
    def last_file(self):
        return self._filename_iter.last_name

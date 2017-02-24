import os
import h5py
import numpy as np

from .base import DataSetImages
from ..utils.cells import Sampler
""" Forced to use patches. """

# TODO : Implementation


class Flickr25k(DataSetImages):

    def __init__(self, **kwargs):
        super(Flickr25k, self).__init__(**kwargs)
        if self._file_data is None:
            self._file_data = os.environ.get('PATH_FLICKR25K')
        self._fin = None
        self._sampler = None

    def __enter__(self):
        self._fin = h5py.File(self._file_data, 'r')
        data_keys = list(self._fin.keys())
        self._nb_datas = len(data_keys)
        if self._is_train:
            self._sampler = Sampler(datas=data_keys, is_shuffle=True)
        else:
            self._sampler = Sampler(datas=data_keys, is_shuffle=False)
        return self

    def visualize(self, sample):
        image = super(Flickr25k, self).visualize(sample)  
        if self._is_batch:
            image = [np.uint8(im) for im in image]
        else:
            image = np.uint8(image)
        return image

    def _sample_data_label_weight(self):
        image = np.array(self._fin[next(self._sampler)[0]], dtype=np.float32)
        if self._is_crop:
            image = self._crop(image)
        if self._is_gray:
            image = np.mean(image, axis=-1, keepdims=True)
        if self._is_norm:
            image /= self._norm_c
        if self._is_down_sample:
            label = np.array(image, dtype=np.float32)
            image = self._downsample(image)
        else:
            label = image
        return image, label, 1.0

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
        if self._fn_data is None:
            self._fn_data = os.environ.get('PATH_FLICKR25K')
        self._fin = None
        self._sampler = None

    def __enter__(self):
        self._fin = h5py.File(self._fn_data, 'r')
        data_keys = list(self._fin.keys())
        self._nb_datas = len(data_keys)
        if self._is_train:
            self._sampler = Sampler(datas=data_keys, is_shuffle=True)
        else:
            self._sampler = Sampler(datas=data_keys, is_shuffle=False)

    def __exit__(self):
        self._fin.close()

    def _sample_data_label_weight(self):
        image = np.array(self._fin[next(self._sampler)], dtype=np.float32) / 256.0
        if self._is_4d:
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        if self._is_gray:
            image = np.mean(image, axis=-1)
            image = image.reshape(list(image.shape)+[1])
        return image, image, 1.0

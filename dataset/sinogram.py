import os
import h5py
import numpy as np

from .base import DataSetImages
from ..utils.cells import Sampler
from ..utils.tensor import down_sample_nd
from ..utils.general import with_config
""" Forced to use patches. """

# TODO : Implementation


class Sinograms(DataSetImages):
    @with_config
    def __init__(self, settings=None, is_padding=False, **kwargs):
        super(Sinograms, self).__init__(**kwargs)
        self._is_padding = settings.get('is_padding', is_padding)
        self._c.update({'is_padding'})        
        if self._fn_data is None:
            self._fn_data = os.environ.get('PATH_SINOGRAMS')
        self._fin = None
        self._sampler = None
        self._datas = None
        
    def __enter__(self):
        self._fin = h5py.File(self._fn_data, 'r')
        self._datas = self._fin['sinograms']
        self._nb_datas = self._datas.shape[0]
        if self._is_train:
            self._sampler = Sampler(datas=list(
                range(self._nb_datas)), is_shuffle=True)
        else:
            self._sampler = Sampler(datas=list(
                range(self._nb_datas)), is_shuffle=False)

    def __exit__(self):
        self._fin.close()

    def _sample_data_label_weight(self):
        sino = np.array(self._datas[next(self._sampler)], dtype=np.float32)
        if self._is_4d:
            sino = sino.reshape(([1] + list(sino.shape) + [1])
        if self._is_down_sample:
            label=np.array(sino)
            sino=down_sample_nd(sino, self._down_sample_ratio)
            return sino, label, 1.0
        else:
            return sino, sino, 1.0

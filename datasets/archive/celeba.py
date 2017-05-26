import os
import h5py
from ..utils.cells import Sampler
from .base import DataSetImages


class Celeba(DataSetImages):
    def __init__(self, **kwargs):
        DataSetImages.__init__(self, **kwargs)
        self._is_filedata = True
        self._file_data = os.path.join(
            os.environ.get('PATH_DATASETS'), 'celeba.h5')

    def __enter__(self):
        self._is_init = True
        if self._is_filedata:
            self._fin = h5py.File(self._file_data, 'r')
            self._dataset = self._fin.get('images')
            self._nb_examples = self._dataset.shape[0]
            if self._is_train:
                self._sampler = Sampler(datas=range(
                    self._nb_examples), is_shuffle=True)
            else:
                self._sampler = Sampler(datas=range(
                    self._nb_examples), is_shuffle=False)
        return self

import os
import h5py
import numpy as np

from .base import DataSetImages


class Celeba(DataSetImages):
    def __init__(self, **kwargs):
        DataSetImages.__init__(self, **kwargs)
        self._is_filedata = True
        if self._file_data is None:
            self._file_data = os.path.join(
                os.environ.get('PATH_DATASETS'), 'celeba.h5')

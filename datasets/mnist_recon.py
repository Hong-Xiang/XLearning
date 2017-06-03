import os
import pathlib
import numpy

from ..utils.general import with_config
from .base import DataSetImages, PATH_DATASETS

class MNISTRecon(DataSetImages):
    @with_config
    def __init__(self, **kwargs):
        DataSetImages.__init__(self,
                               crop_shape=[28, 28],
                               is_down_sample=False,
                               **kwargs)
        self.params['keys'] = ['data', 'label', 'idx']
        self.params.update_short_cut()

    def initialize(self):
        super(MNISTRecon, self).initialize()
        if self.p.mode == 'train':
            self.labels = self.fin['y_train']
        else:
            self.labels = self.fin['y_test']

    def _sample_single(self):
        """ read from dataset HDF5 file and perform necessary preprocessings """
        ss = DataSetImages._sample_single(self)
        image = ss['data']
        idx = ss['idx']
        label = self.labels[idx]  
        return {'data': image, 'label': label, 'idx': idx}

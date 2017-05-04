import os
import pathlib
import numpy

from ..utils.general import with_config
from .base import DataSetImages, PATH_DATASETS

class MNISTRecon(DataSetImages):
    @with_config
    def __init__(self, **kwargs):
        DataSetImages.__init__(self, **kwargs)
        if self.mode == 'train':
            self.data_key = 'x_train'
        else:
            self.data_key = 'x_test'
        self.labels = None
        if self.file_data is None:
            self.file_data = pathlib.Path(PATH_DATASETS) / 'mnist.h5'

    def initialize(self):
        super(MNISTRecon, self).initialize()
        if self.mode == 'train':
            self.labels = self.fin['y_train']
        else:
            self.labels = self.fin['y_test']

    def _load_sample(self):
        idx = next(self._sampler)[0]
        image = numpy.array(self._dataset[idx], dtype=numpy.float32)
        if len(image.shape) == 2:
            image = image.reshape([1, image.shape[0], image.shape[1]])
        label = numpy.array(self.labels[idx], dtype=numpy.int32)
        
        return image, label

    def _sample_single(self):
        """ read from dataset HDF5 file and perform necessary preprocessings """
        image, label = self._load_sample()
        if self.is_norm:
            image = self.norm(image)

        return {'data': image, 'label': label}

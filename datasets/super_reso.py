import os
import pathlib
import numpy

from ..utils.general import with_config
from .base import DataSetImages, PATH_DATASETS


class SuperResolutions(DataSetImages):
    @with_config
    def __init__(self,
                 data_name=None,
                 **kwargs):
        DataSetImages.__init__(self, **kwargs)
        self.data_name = data_name
        self.pp_dict.update({
            'data_name': self.data_name
        })
        if self.file_data is None:
            if self.data_name == 'mnist':
                self.file_data = str(pathlib.Path(PATH_DATASETS) / 'mnist.h5')
                if self.mode == 'train':
                    self.data_key = 'x_train'
                else:
                    self.data_key = 'x_test'

            elif self.data_name == 'shep':
                self.file_data = str(pathlib.Path(
                    PATH_DATASETS) / 'shepplogan_sinograms.h5')
                self.data_key = "sinograms"
            else:
                raise ValueError('Unknown data_name {}.'.format(data_name))
            self.pp_dict.update({
                "file_data": self.file_data,
                "data_key": self.data_key
            })

    def _sample_single(self):
        """ read from dataset HDF5 file and perform necessary preprocessings """
        ss = super(SuperResolutions, self)._sample_single()
        image = ss['data']
        # Down sample
        max_down_sample = max(self.data_down_sample, self.label_down_sample)
        if self.is_down_sample:
            images = []
            images.append(image)
            for i in range(max_down_sample):
                image = self.downsample(image)
                images.append(image)
            data = images[self.data_down_sample]
            label = images[self.label_down_sample]
        return {'data': data, 'label': label}

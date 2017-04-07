import os
import h5py
import numpy as np

from .base import DataSetImages
from ..utils.cells import Sampler
from ..utils.tensor import down_sample_nd
from ..utils.general import with_config


class Sinograms(DataSetImages):
    """ Load HDF5 dataset of sinograms.
    Sinograms are saved in dataset = fin.get('sinograms'), with shape (nb_sinograms, nb_sensors, nb_views).
    """
    @with_config
    def __init__(self,
                 sino_name='shepplogan',
                 **kwargs):
        super(Sinograms, self).__init__(**kwargs)
        if self._file_data is None:
            self._file_data = os.path.join(os.environ.get('PATH_DATASETS'), sino_name+'_sinograms.h5')
        self._fin = None
        self._sampler = None        
    
    
    def _load_sample(self):
        idx = next(self._sampler)[0]
        image = np.array(self._dataset[idx])
        image = image[:, :360, :]
        image += 1
        image = np.log(image)

        if len(image.shape) == 2:
            image = image.reshape([image.shape[0], image.shape[1], 1])
        return image

    def __enter__(self):
        self._fin = h5py.File(self._file_data, 'r')
        self._dataset = self._fin['sinograms']
        self._nb_datas = self._dataset.shape[0]
        if self._is_train:
            self._sampler = Sampler(datas=list(
                range(self._nb_datas)), is_shuffle=True)
        else:
            self._sampler = Sampler(datas=list(
                range(self._nb_datas)), is_shuffle=False)
        return self

    # def _sample_data_label_weight(self):
    #     sino = np.array(self._datas[next(self._sampler)[
    #                     0], :, :], dtype=np.float32)
    #     sino = sino.reshape(sino.shape[0], sino.shape[1], 1)
    #     if self._is_crop:
    #         sino = self._crop(sino)
    #     if self._is_norm:
    #         sino /= self._norm_c
    #     if self._is_down_sample:
    #         label = np.array(sino, dtype=sino.dtype)
    #         sino = self._downsample(sino)
    #     return sino, label, 1.0

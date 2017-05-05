import pathlib
import h5py
import numpy as np
from ..utils.general import with_config
from ..utils.cells import Sampler
from .base import DataSetBase, PATH_DATASETS
import random

class CalibrationDataSet(DataSetBase):
    @with_config
    def __init__(self,
                 is_good=False,
                 is_agu=True,
                 with_z=False,
                 with_y=True,
                 **kwargs):
        DataSetBase.__init__(self, **kwargs)
        self.is_good = is_good
        self.is_agu = is_agu
        self.with_z = with_z
        self.with_y = with_y
        if self.is_good:
            self.data_key = {'data': 'evt_good', 'label': 'inc_good'}
        else:
            self.data_key = {'data': 'evt_all', 'label': 'inc_all'}
        if self.file_data is None:
            self.file_data = str(
                (pathlib.Path(PATH_DATASETS) / 'cali.h5').absolute())
        self.pp_dict.update({
            'is_good': self.is_good,
            'data_key': self.data_key,
            'file_data': self.file_data,
            'with_z': self.with_z,
            'with_y': self.with_y,
            'is_agu': self.is_agu
        })

    def initialize(self):
        self.fin = h5py.File(self.file_data, 'r')
        self.data = self.fin[self.data_key['data']]
        self.label = self.fin[self.data_key['label']]
        nb_total = self.data.shape[0]

        nb_train = nb_total // 5 * 4
        if self.mode == 'train':
            self.nb_examples = nb_train
            self.sampler = Sampler(list(range(nb_train)))
        else:
            self.nb_examples = nb_total - nb_train
            self.sampler = Sampler(list(range(nb_train, nb_total)))
        self.pp_dict.update({
            'nb_examples': self.nb_examples
        })
        super(CalibrationDataSet, self).initialize()

    def finalize(self):
        self.fin.close()
        super(CalibrationDataSet, self).finalize()

    def _sample_single(self):
        """ interface of sampling """
        while True:
            idx = next(self.sampler)[0]
            data = self.data[idx, ...]
            data = data.reshape([10, 10])
            label = self.label[idx, ...]
            if self.is_agu:
                isflip = random.randint(0, 1)
                if isflip == 1:
                    data = data[9::-1, :]
                    label[1] = - label[1]
                isflip = random.randint(0, 1)
                if isflip == 1:
                    data = data[:, 9::-1]
                    label[0] = - label[0]
                istrans = random.randint(0, 1)
                if istrans == 1:
                    data = data.T
                    label[0], label[1] = label[1], label[0]
            data = data.reshape([1, 10, 10])
            if not self.with_z:
                label = label[:2]
            if not self.with_y:
                label = label[:1]
            total = np.sum(data)
            if total > 5500:
                break
        data = self.norm(data)
        return {'data': data, 'label': label}

import pathlib
import h5py
import numpy as np
from ..utils.general import with_config
from ..utils.cells import Sampler
from .base import DataSetBase, PATH_DATASETS


class CalibrationDataSet(DataSetBase):
    @with_config
    def __init__(self,
                 is_good=False,
                 with_z=False,
                 **kwargs):
        DataSetBase.__init__(self, **kwargs)
        self.is_good = is_good
        self.with_z = with_z
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
            'with_z': self.with_z
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
            data = data.reshape([1, 10, 10])
            label = self.label[idx, ...]
            if not self.with_z:
                label = label[:2]
            total = np.sum(data)
            if total > 5500:
                break
        data = self.norm(data)
        return {'data': data, 'label': label}

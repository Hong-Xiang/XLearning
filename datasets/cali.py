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
                 is_mix=False,
                 with_z=False,
                 with_y=True,
                 **kwargs):
        DataSetBase.__init__(self, **kwargs)
        self.is_good = is_good
        self.is_agu = is_agu
        self.is_mix = is_mix
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
        self.data = None
        self.label = None

    def augmentation(self, data, label):
        if self.is_agu:
            datanew = []
            labelnew = []
            nb_samples = data.shape[0]
            for i in range(nb_samples):
                d0 = data[i, ...].reshape([10, 10])
                l0 = label[i, ...]
                datanew.append(d0)
                labelnew.append(l0)
                l1 = np.array(l0)
                d1 = d0[9::-1, :]
                l1[1] = - l0[1]
                datanew.append(d1)
                labelnew.append(l1)
                l2 = np.array(l0)
                d2 = d0[:, 9::-1]
                l2[0] = - l0[0]
                datanew.append(d2)
                labelnew.append(l2)
                d3 = np.array(d0)
                d3 = d3.T
                l3 = np.array(l0)
                l3[0], l3[1] = l0[1], l0[0]
                datanew.append(d3)
                labelnew.append(l3)
                l4 = np.array(l3)
                d4 = d3[9::-1, :]
                l4[1] = - l3[1]
                datanew.append(d4)
                labelnew.append(l4)
                l5 = np.array(l3)
                d5 = d3[:, 9::-1]
                l5[0] = - l5[0]
                datanew.append(d5)
                labelnew.append(l5)

            idx = list(range(nb_samples * 6))
            random.shuffle(idx)
            datanew_shuffle = []
            labelnew_shuffle = []
            for i in idx:
                datanew_shuffle.append(datanew[i])
                labelnew_shuffle.append(labelnew[i])
            datanew = np.array(datanew_shuffle)
            labelnew = np.array(labelnew_shuffle)
        else:
            datanew = np.array(data)
            labelnew = np.array(label)
        return datanew, labelnew

    def initialize(self):
        with h5py.File(self.file_data, 'r') as fin:
            data = np.array(fin[self.data_key['data']])
            label = np.array(fin[self.data_key['label']])
        if self.is_mix:
            self.data, self.label = self.augmentation(data, label)
        else:
            nb_total = data.shape[0]
            nb_train = nb_total // 5 * 4    
            data_train = data[:nb_train, ...]
            data_test = data[nb_train:, ...]
            label_train = label[:nb_train, ...]
            label_test = label[nb_train:, ...]
            data_train, label_train = self.augmentation(data_train, label_train)
            data_test, label_test = self.augmentation(data_test, label_test)
            self.data = np.concatenate([data_train, data_test])
            self.label = np.concatenate([label_train, label_test])
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

    def _sample_single(self):
        """ interface of sampling """
        while True:
            idx = next(self.sampler)[0]
            data = self.data[idx, ...]
            label = self.label[idx, ...]
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

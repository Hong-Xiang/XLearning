""" Raw data loader, with filters """
import numpy as np
from xlearn.utils.options import json_config, auto_configs
from xlearn.utils.cells import Sampler

class IndexSampler:
    def __init__(self, is_shuffle=True, idx_file=None, idx_range=None):
        if idx_file is not None:
            self.idx = np.load(idx_file)
        else:
            self.idx = list(range(idx_range[0], idx_range[1]))
        self.sampler = Sampler(self.idx, is_shuffle=is_shuffle)

    def _generate(self):
        return next(self.sampler)[0]
    def sample(self):
        return self._generate()
    def __next__(self):
        return self.sample()

class Loader:
    def __init__(self, datasets, idx_sampler):
        if not isinstance(datasets, dict):
            fmt = r"Dataset type must be {type_dict}, given {type_data}."
            msg = fmt.format(type_dict=dict, type_data=type(datasets))
            raise TypeError(msg)
        self.datasets = datasets
        self.idx_sampler = idx_sampler

    def _generate(self):
        idx = next(self.idx_sampler)
        out = {k: np.array(self.datasets[k][idx, ...]) for k in self.datasets}
        out.update({'idx': idx})
        return out

    def sample(self):
        return self._generate()            

    def __next__(self):
        return self.sample()


class NoneZeroLoader(Loader):
    @auto_configs(exclude=('datasets', 'idx_sampler'))
    def __init__(self, fra_none_zero=0.1, keys=None, keys_ign=('idx', ), nb_max_try=100, datasets=None, idx_sampler=None):
        super(NoneZeroLoader, self).__init__(datasets=datasets)

    def _generate(self):
        valid = False        
        nb_try = 0        
        while not valid:
            data_dict = super(NoneZeroLoader, self)._generate()
            keys2chk = list(datasets.keys) if self.c.keys is None else list(
                self.c.keys)            
            valid = True
            for k in keys2chk:
                if k in self.c.keys_ign:
                    continue
                size = data_dict[k].size
                nnz = np.sum(data_dict[k] > 0)
                size, nnz = np.float64(size), np.float64(nnz)
                if nnz / size < self.c.fra_none_zero:
                    valid = False
                    break
            nb_try += 1
            if nb_try >= nb_max_try:
                raise ValueError("Maximum try time exceeded.")
        return data_dict

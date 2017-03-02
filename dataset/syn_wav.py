import numpy as np

from .base import DataSetBase
from ..utils.general import with_config

class SynWave(DataSetBase):
    @with_config
    def __init__(self,
                 x_min=0.0,
                 x_max=4 * np.pi,
                 a_min=1.0,
                 a_max=3.0,
                 k_min=1.0,
                 k_max=2.0,
                 x_dim=100,
                 is_bin=False,
                 is_cata=False,
                 nb_cata=100,
                 settings=None,
                 **kwargs):

        super(SynWave, self).__init__(**kwargs)
        self._settings = settings
        self._x_min = self._update_settings('x_min', x_min)
        self._x_max = self._update_settings('x_max', x_max)
        self._a_min = self._update_settings('a_min', a_min)
        self._a_max = self._update_settings('a_max', a_max)
        self._k_min = self._update_settings('k_min', k_min)
        self._k_max = self._update_settings('k_max', k_max)
        self._x_dim = self._update_settings('x_dim', x_dim)
        self._is_bin = self._update_settings('is_bin', is_bin)
        self._is_cata = self._update_settings('is_cata', is_cata)
        self._nb_cata = self._update_settings('nb_cata', nb_cata)
        self._x = np.linspace(self._x_min, self._x_max, self._x_dim)

    def _sample_data_label_weight(self):
        a = np.random.uniform(self._a_min, self._a_max)
        k = np.random.uniform(self._k_min, self._k_max)
        p = np.random.uniform(0.0, 2 * np.pi)
        x = a * np.sin(k * self._x + p)
        data = np.array(x, dtype=x.dtype)
        label = np.array(x, dtype=x.dtype)
        if is_bin:
            x[x < 0] = -a
            x[x >= 0] = a
            label[label < 0] = -a
            label[lable >= 0] = a
        if is_cata:            
            ids = np.zeros(shape=x.shape, )
            for i in range(self._nb_cata):
                
                
        return (data, label, 1.0)

    def visualize(self, sample, **kwargs):
        if self._is_batch:
            return sample.reshape([self._batch_size * self._x_dim])
        else:
            return sample
    @property
    def x(self):
        return self._x

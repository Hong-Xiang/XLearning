import numpy as np

from .base import DataSetBase


class SynWave(DataSetBase):

    def __init__(self,
                 x_min=0.0,
                 x_max=4 * np.pi,
                 a_min=1.0,
                 a_max=3.0,
                 k_min=1.0,
                 k_max=2.0,
                 x_dim=100,
                 **kwargs):

        super(SynWave, self).__init__(**kwargs)
        self._x_min = self._settings.get('x_min', x_min)
        self._x_max = self._settings.get('x_max', x_max)
        self._a_min = self._settings.get('a_min', a_min)
        self._a_max = self._settings.get('a_max', a_max)
        self._k_min = self._settings.get('k_min', k_min)
        self._k_max = self._settings.get('k_max', k_max)
        self._x_dim = self._settings.get('x_dim', x_dim)
        self._x = np.linspace(self._x_min, self._x_max, self._x_dim)

    def _sample_data_label_weight(self):
        a = np.random.uniform(self._a_min, self._a_max)
        k = np.random.uniform(self._k_min, self._k_max)
        p = np.random.uniform(0.0, 2 * np.pi)
        x = a * np.sin(k * self._x + p)
        return (x, x, 1.0)

    def visualize(self, sample, **kwargs):
        if self._is_batch:
            return sample.reshape([self._batch_size * self._x_dim])
        else:
            return sample
    @property
    def x(self):
        return self._x

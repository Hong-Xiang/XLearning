""" Mixture of Gaussian data """
import numpy as np
import random
from .base import DataSetBase
from ..utils.general import with_config


class MoG(DataSetBase):
    """ generate data from mixture of Gaussian model """
    @with_config
    def __init__(self,
                 sigma=1,
                 data_dim=2,
                 nb_center=8,
                 mus=None,
                 settings=None,
                 **kwargs):
        super(MoG, self).__init__(**kwargs)
        self._settings = settings
        self._data_dim = self._update_settings('data_dim', data_dim)
        self._nb_center = self._update_settings('nb_center', nb_center)
        self._mus = self._update_settings('mus', mus)
        self._sigma = self._update_settings('sigma', sigma)
        if self._mus is not None:
            self._nb_center, self._data_dim = mus.shape
        else:
            self._mus = self.randunif(-1, 1, (self._data_dim, ))

        self._condition = None

    def set_condition(self, value=None):
        """ set condition """
        self._condition = value

    def _sample_data_label_weight(self):
        if self._condition is None:
            id_center = self.randint(0, self._nb_center - 1)
        else:
            id_center = self._condition

        y = self._mus[id_center, :]
        y = y.reshape((self._data_dim,))
        x = self.randnorm(y, sigma=self._sigma)
        return (x, y, 1.0)

class Wave(DataSetBase):
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

        super(Wave, self).__init__(**kwargs)
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
        a = self.rand_unif(self._a_min, self._a_max, (self.batch_size, 1))
        k = self.rand_unif(self._k_min, self._k_max, (self.batch_size, 1))
        p = self.rand_unif(0.0, 2 * np.pi, (self.batch_size, 1))
        x = a * np.sin(k * self._x + p)
        data = np.array(x, dtype=x.dtype)
        label = np.array(x, dtype=x.dtype)
        if self._is_bin:
            x[x < 0] = -a
            x[x >= 0] = a
            label[label < 0] = -a
            label[lable >= 0] = a                
        return (data, label, 1.0)

    def visualize(self, sample, **kwargs):
        return sample.reshape([-1])

    @property
    def x(self):
        return self._x

class Syn4(DataSetBase):
    """ A synthetic example with four kind of clean inputs:
    +-+-+       +-+-+       +-+-+       +-+-+
    |1|0|       |0|1|       |0|0|       |0|0|
    +-+-+       +-+-+       +-+-+       +-+-+
    |0|0|       |0|0|       |0|1|       |1|0|
    +-+-+       +-+-+       +-+-+       +-+-+
    (1,0;0,0)   (0,1;0,0)   (0,0;0,1)   (0,0;1,0)
    """

    def __init__(self, noise_type='gaussian', noise_scale=1.0, fixed_z=False, z_value=0, **kwargs):
        super(Syn4, self).__init__(**kwargs)
        self._noise_type = noise_type
        self._noise_scale = noise_scale
        self._fixed_z = fixed_z
        self._z_value = z_value

    def set_z(self, v):
        self._z_value = v
        self._fixed_z = True

    def unset_z(self):
        self._fixed_z = False

    def _sample_data_label_weight(self):
        if self._fixed_z:
            z = self._z_value
        else:
            z = random.randint(0, 3)
        if z == 0:
            y = numpy.array([1, 0, 0, 0])
        elif z == 1:
            y = numpy.array([0, 1, 0, 0])
        elif z == 2:
            y = numpy.array([0, 0, 0, 1])
        elif z == 3:
            y = numpy.array([0, 0, 1, 0])
        if self._noise_scale == 0.0:
            n = numpy.zeros(y.shape)
        elif self._noise_type == 'gaussian':
            n = numpy.random.normal(0, self._noise_scale, (4, ))
        elif self._noise_type == 'uniform':
            n = numpy.random.uniform(-self._noise_scale,
                                     self._noise_scale, (4, ))
        x = y + n
        return (x, y, 1.0)

    def visualize(self, image, **kwargs):
        image = image.reshape((-1, 2, 2))
        if image.shape[0] == 1:
            img = image.reshape(image, (2, 2))
        else:
            img = []
            for i in range(image.shape[0]):
                img.append(image[i, :, :])
        return img
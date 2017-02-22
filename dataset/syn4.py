""" A synthetic example with four kind of clean inputs:
+-+-+       +-+-+       +-+-+       +-+-+
|1|0|       |0|1|       |0|0|       |0|0|
+-+-+       +-+-+       +-+-+       +-+-+
|0|0|       |0|0|       |0|1|       |1|0|
+-+-+       +-+-+       +-+-+       +-+-+
(1,0;0,0)   (0,1;0,0)   (0,0;0,1)   (0,0;1,0)
"""

import random
import numpy
from xlearn.dataset.base import DataSetBase


class Syn4(DataSetBase):
    """ dataset of synthetic4 """

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

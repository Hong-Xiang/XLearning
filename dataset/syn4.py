""" A synthetic example with four kind of clean inputs:
+-+-+       +-+-+       +-+-+       +-+-+
|1|1|       |1|1|       |0|1|       |1|0|
+-+-+       +-+-+       +-+-+       +-+-+
|1|0|       |0|1|       |1|1|       |1|1|
+-+-+       +-+-+       +-+-+       +-+-+
(1,1,1,0)   (1,1,0,1)   (0,1,1,1)   (1,0,1,1)
"""

from xlearn.dataset.base import DataSetBase
import random
import numpy


class Syn4(DataSetBase):

    def __init__(self, noise_type='gaussian', noise_scale=1.0, **kwargs):
        super(Syn4, self).__init__(**kwargs)
        self._noise_type = noise_type
        self._noise_scale = noise_scale

    def _sample_data_label_weight(self):
        z = random.randint(0, 4)
        if z == 0:
            y = numpy.array([1, 1, 1, 0])
        elif z == 1:
            y = numpy.array([1, 1, 0, 1])
        elif z == 2:
            y = numpy.array([0, 1, 1, 1])
        elif z == 3:
            y = numpy.array([1, 0, 1, 1])
        if self._noise_type == 'gaussian':
            n = numpy.random.normal(0, self._noise_scale, [4])
        elif self._noise_type == 'uniform':
            n = numpy.random.uniform(-self._noise_scale,
                                     self._noise_scale, [4])
        x = y + n
        return (x, y, 1.0)

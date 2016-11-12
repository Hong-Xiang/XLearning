
from __future__ import absolute_import, division, print_function
import numpy as np
import random
from six.moves import xrange

class DataSetOneOverX(object):
    def __init__(self, batch_size=256):
        self._batch_size = batch_size

    def next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self._batch_size
        x = np.random.sample(size=[batch_size, 1]) + 0.1
        y = 1 / x
        return x, y
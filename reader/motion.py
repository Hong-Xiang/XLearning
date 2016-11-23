
from __future__ import absolute_import, division, print_function
import numpy as np
import random
from six.moves import xrange

class DataSetOneOverX(object):
    def __init__(self, batch_size=256, frames=256):
        self._batch_size = batch_size
        self._frames = frames

    def next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self._batch_size
        t = np.linspace(0, 1, self._frames)
        output = np.zeros(batch_size, 3, self._frames)
        for i in xrange(batch_size):
            type = random.randint(0, 3)
            if type == 1:
                "linear"
                for k in xrange(3):
                    a = random.uniform(-1, 1)
                    b = random.uniform(-1, 1)
                    output[i, k, :] = a * t + b
            if type == 2:
                "circle"
                r = random.uniform(0, 1)
                a = random.uniform(-1, 1)
                b = random.uniform(-1, 1)
                output[i, 0, :] = np.cos((t+a)*np.pi)*r
                output[i, 1, :] = np.sin((t+a)*np.pi)*r
                output[i, 2, :] = t*(1-t)*b
        return x, y
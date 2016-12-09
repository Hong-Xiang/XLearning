
from __future__ import absolute_import, division, print_function
import numpy as np
import random
from six.moves import xrange
from xlearn.reader.base import DataSet

raise DeprecationWarning


class DataSetOneOverX(object):

    def __init__(self, batch_size=256):

        self._batch_size = batch_size

    def next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self._batch_size
        x = np.random.sample(size=[batch_size, 1]) + 0.1
        y = 1 / x
        return x, y


class DataSetFx(DataSet):
    """DataSetOfGeneralF(x)
    """

    def __init__(self, batch_size=256, foo=None, **kwargs):
        super(DataSetFx, self).__init__(batch_size=batch_size,
                                        shape_i=[batch_size, 1],
                                        shape_o=[batch_size, 1],
                                        is_pad=True,
                                        is_cache=True,
                                        epoch_max=None,
                                        **kwargs)

        if foo is None:
            self._foo = lambda x: 1 / x
        else:
            self._foo = foo

    def _gather_paras(self, dataset_type):
        super(DataSetFx, self)._gather_paras(dataset_type)
        self._xmin = self._paras['xmin']
        self._xmax = self._paras['xmax']

    def _single_sample(self):
        x = np.random.sample(size=[1, 1]) * \
            (self._xmax - self._xmin) + self._xmin
        y = self._foo(x)
        if self._is_train_or_test or self._is_cache:
            return x, y
        else:
            return x

    @property
    def foo(self):
        return self._foo

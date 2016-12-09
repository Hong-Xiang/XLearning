"""data sets of general f(x)
"""
import numpy as np
from xlearn.reader.base import DataSet


class DataSetFx(DataSet):
    """data set of gengeral f(x)
    """

    def __init__(self, batch_size=256, func=None, **kwargs):
        super(DataSetFx, self).__init__(batch_size=batch_size,
                                        shape_i=[batch_size, 1],
                                        shape_o=[batch_size, 1],
                                        is_pad=True,
                                        is_cache=True,
                                        epoch_max=None,
                                        **kwargs)

        if func is None:
            self._func = lambda x: 1 / x
        else:
            self._func = func

    def _gather_paras(self, dataset_type):
        super(DataSetFx, self)._gather_paras(dataset_type)
        self._xmin = self._paras['xmin']
        self._xmax = self._paras['xmax']

    def _single_sample(self):
        x = np.random.sample(size=[1, 1]) * \
            (self._xmax - self._xmin) + self._xmin
        y = self._func(x)
        if self._is_train_or_test or self._is_cache:
            return x, y
        else:
            return x

    @property
    def func(self):
        """funtion to x"""
        return self._func

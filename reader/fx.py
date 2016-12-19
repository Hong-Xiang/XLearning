"""data sets of general f(x)
"""
import numpy as np
import logging
from xlearn.reader.base import DataSet


class DataSetFx(DataSet):
    """data set of gengeral f(x)
    Generate uniform random x in interval (xmin, xmax)
    Generate y = self.func(x)
    """

    def __init__(self, batch_size=None, func=None, **kwargs):
        super(DataSetFx, self).__init__(batch_size=batch_size,
                                        shape_i=[1],
                                        shape_o=[1],
                                        is_pad=True,
                                        is_cache=True,
                                        epoch_max=None,
                                        **kwargs)

        if batch_size is not None:
            self._batch_size = batch_size

    def _get_default_paras(self):
        paras_def = super(DataSetFx, self)._get_default_paras()
        paras_def.update({'func': lambda x: 1 / x})
        paras_def.update({'xmin': 1})
        paras_def.update({'xmax': 10})
        return paras_def

    def _gather_paras_common(self):
        super(DataSetFx, self)._gather_paras_common()
        self._xmin = self._paras['xmin']
        self._xmax = self._paras['xmax']
        self._func = self._paras['func']

    def _sample(self):
        x_in = np.random.sample(size=[1, 1]) * \
            (self._xmax - self._xmin) + self._xmin
        y_out = self._func(x_in)
        return x_in, y_out

    @property
    def func(self):
        """funtion to x"""
        return self._func

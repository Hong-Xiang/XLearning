"""data sets of general f(x)
"""
import numpy as np
import logging
from xlearn.reader.base import DataSet


class DataSetFx(DataSet):
    """data set of gengeral f(x)
    Generate uniform random x in interval (xmin, xmax)
    Generate y = self.func(c*x)
    is_possion adds Possion noise
    is_gaussion adds Gaussian noise
    """

    def __init__(self, batch_size=None, func=None, **kwargs):
        super(DataSetFx, self).__init__(batch_size=batch_size,
                                        shape_i=[1],
                                        shape_o=[1],
                                        is_pad=True,
                                        is_cache=True,
                                        epoch_max=None,
                                        func=func,
                                        **kwargs)

        if batch_size is not None:
            self._batch_size = batch_size
        

    def _get_default_paras(self):
        paras_def = super(DataSetFx, self)._get_default_paras()
        paras_def.update({'func': lambda x: 1 / x})
        paras_def.update({'xmin': 1})
        paras_def.update({'xmax': 10})
        paras_def.update({'is_possion': False})
        paras_def.update({'is_gaussian': False})
        paras_def.update({'noise_scale': 1.0})
        paras_def.update({'c': 1.0})
        paras_def.update({'func_name': "unknown"})
        paras_def.update({'is_bayes': False})
        return paras_def

    def _gather_paras_common(self):
        super(DataSetFx, self)._gather_paras_common()
        self._xmin = self._paras['xmin']
        self._xmax = self._paras['xmax']
        self._func = self._paras['func']
        self._c = self._paras['c']
        self._is_possion = self._paras['is_possion']
        self._is_gaussian = self._paras['is_gaussian']
        self._is_uniform = self._paras['is_uniform']
        self._noise_scale = self._paras['noise_scale']
        self._func_name = self._paras['func_name']

        if self._func_name != "unknown":
            if self._func_name == "one_over_x":
                self._func = lambda x: 1/x
            if self._func_name == "sin":
                self._func = lambda x: np.sin(x)
            if self._func_name == "cos":
                self._func = lambda x: np.cos(x)
            if self._func_name == "linear":
                self._func = lambda x: x
            if self._func_name == "10sin2":
                self._func = lambda x: 10 * np.sin(x) * np.sin(x)

    def _sample(self):
        x_in = np.random.sample(size=[1, 1]) * \
            (self._xmax - self._xmin) + self._xmin
        y = self._func(self._c * x_in)
        if self._is_possion:
            y = np.random.poisson(lam=y, size=[1, 1])
        if self._is_gaussian:
            y = y + np.random.normal(0.0, self._noise_scale, y.shape)
        if self._is_uniform:            
            y = y + np.random.uniform(-self._noise_scale, +self._noise_scale,y.shape)
        return x_in, y

    @property
    def func(self):
        """funtion to x"""
        return self._func

    @property
    def xmin(self):
        return self._xmin
    
    @property
    def xmax(self):
        return self._xmax
    
    @property
    def c(self):
        return self._c
    
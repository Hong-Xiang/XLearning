""" Warp for mnist based on keras.datasets.mnist
"""

import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

from .base import DataSetBase
from ..utils.general import with_config
from ..utils.cells import Sampler

#TODO: Impl is_bin

class MNIST(DataSetBase):
    """ MNIST dataset based on kears.datasets.mnist. With enhanced methods.
    """

    def __init__(self, is_unsp=True, is_train=True, is_noise=False, is_4d=False, noise_scale=0.0, noise_type='poisson',
                 is_norm=False,
                 is_flatten=False,
                 is_bin=False,
                 **kwargs):
        super(MNIST, self).__init__(is_train=is_train, is_noise=is_noise,
                                    noise_scale=noise_scale, noise_type=noise_type,
                                    is_finite=True,
                                    is_norm=is_norm,
                                    is_flatten=is_flatten,
                                    is_unsp=is_unsp,
                                    is_4d=is_4d,
                                    is_bin=is_bin,
                                    **kwargs)
        self._is_train = self._settings['is_train']
        self._is_noise = self._settings['is_noise']
        self._noise_scale = self._settings['noise_scale']
        self._noise_type = self._settings['noise_type']
        self._is_unsp = self._settings['is_unsp']
        self._is_norm = self._settings['is_norm']
        self._is_flatten = self._settings['is_flatten']
        self._is_4d = self._settings['is_4d']
        self._is_bin = self._settings['is_bin']

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        if self._is_train:
            self._data = x_train
            self._label = y_train
        else:
            self._data = x_test
            self._label = y_test

        self._combine = [(d, l) for d, l in zip(self._data, self._label)]

        self._sampler = Sampler(datas=self._combine)    

    def visualize(self, image, **kwargs):
        if self._is_flatten or self._is_4d:
            image = image.reshape([-1, 28, 28])
        output = []
        if self._is_batch:
            for i in range(image.shape[0]):
                output.append(image[i, :, :])
            image = output
        return image

    def _sample_data_label_weight(self):
        sample = next(self._sampler)
        sample = sample[0]
        image = sample[0]
        digit = sample[1]
        if self._is_norm:
            image = image / 256.0        
        if self._is_flatten:
            image = image.reshape([np.prod(image.shape)])

        if self._is_unsp:
            label = np.copy(image)
        else:
            label = digit
        if self._is_noise:
            if self._noise_type == 'poisson':
                image = np.random.poisson(
                    image / self._noise_scale).astype(np.float32)
                image *= (self._noise_scale / 2.0)
            elif self._noise_type == "gaussian":
                args = image.shape
                noise = np.random.randn(*args) * self._noise_scale
                image = image + noise
        if self._is_4d:
            image = image.reshape((28, 28, 1))
            if self._is_unsp:
                label = label.reshape((28, 28, 1))    
        return (image, label, 1.0)

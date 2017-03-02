""" Warp for mnist based on keras.datasets.mnist
"""

import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

from .base import DataSetBase, DataSetImages
from ..utils.general import with_config
from ..utils.cells import Sampler
from ..utils.tensor import down_sample_nd


class MNIST(DataSetBase):
    """ MNIST dataset based on kears.datasets.mnist. With enhanced methods.
    """
    @with_config
    def __init__(self,
                 is_unsp=True,
                 is_4d=False,
                 is_flatten=False,
                 is_bin=False,
                 is_bernolli=False,
                 is_cata=True,
                 settings=None,
                 **kwargs):
        super(MNIST, self).__init__(**kwargs)
        self._settings = settings
        self._is_unsp = self._update_settings('is_unsp', is_unsp)
        self._is_flatten = self._update_settings('is_flatten', is_flatten)
        self._is_4d = self._update_settings('is_4d', is_4d)
        self._is_bin = self._update_settings('is_bin', is_bin)
        self._is_cata = self._update_settings('is_cata', is_cata)
        self._is_bernolli = self._update_settings('is_bernolli', is_bernolli)

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        if self._is_cata:
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
        if self._is_norm and not self._is_bernolli:
            image += 0.5
        if self._is_flatten or self._is_4d:
            output = image.reshape([-1, 28, 28])
        else:
            output = image
        if self._is_batch:
            output = list(output)
        return output

    def _sample_data_label_weight(self):
        sample = next(self._sampler)
        sample = sample[0]
        image = sample[0]
        digit = sample[1]
        if self._is_bin:
            image[image < 125.1] = 0.0
            image[image >= 125.0] = 255.0
        if self._is_norm:
            image = image / 256.0
            if not self._is_bernolli:
                image -= 0.5
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


class MNISTImage(DataSetImages):
    """ MNIST dataset as images
    """
    @with_config
    def __init__(self,
                 is_bin=False,
                 settings=None,
                 **kwargs):
        super(MNISTImage, self).__init__(**kwargs)
        self._settings = settings
        self._is_bin = self._update_settings('is_bin', is_bin)

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        if self._is_train:
            self._data = x_train
        else:
            self._data = x_test

        self._sampler = Sampler(datas=self._data)

    def _sample_data_label_weight(self):
        sample = next(self._sampler)
        image = sample[0]
        image = image.reshape((28, 28, 1))
        if self._is_bin:
            image[image < 125.1] = 0.0
            image[image >= 125.0] = 255.0
        if self._is_norm:
            image = image / self._norm_c
            image = image - 0.5
        if self._is_crop:
            image = self._crop(image)
        data = image
        label = np.array(image, np.float32)
        if self._is_down_sample:
            data = self._downsample(data)
        if self._is_noise:
            if self._noise_type == 'poisson':
                data = np.random.poisson(
                    data / self._noise_scale).astype(np.float32)
                data *= (self._noise_scale / 2.0)
            elif self._noise_type == "gaussian":
                args = data.shape
                noise = np.random.randn(*args) * self._noise_scale
                data = data + noise
        return (data, label, 1.0)

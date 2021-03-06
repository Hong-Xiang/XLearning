""" Warp for mnist based on keras.datasets.mnist
"""
import os
import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from tqdm import tqdm
import h5py

from .base import DataSetBase, DataSetImages, PATH_DATASETS
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


# class MNIST1D(DataSetBase):
#     """ MNIST dataset based on kears.datasets.mnist. With enhanced methods.
#     """
#     @with_config
#     def __init__(self,
#                  is_unsp=True,
#                  is_bin=False,
#                  is_bernolli=False,
#                  is_cata=True,
#                  settings=None,
#                  **kwargs):
#         super(MNIST1D, self).__init__(**kwargs)
#         self._settings = settings
#         self._is_unsp = self._update_settings('is_unsp', is_unsp)
#         self._is_bin = self._update_settings('is_bin', is_bin)
#         self._is_cata = self._update_settings('is_cata', is_cata)
#         self._is_bernolli = self._update_settings('is_bernolli', is_bernolli)

#         (x_train, y_train), (x_test, y_test) = mnist.load_data()
#         x_train = x_train.astype(np.float32)
#         x_test = x_test.astype(np.float32)
#         if self._is_cata:
#             y_train = to_categorical(y_train, 10)
#             y_test = to_categorical(y_test, 10)
#         if self._is_train:
#             self._data = x_train
#             self._label = y_train
#         else:
#             self._data = x_test
#             self._label = y_test

#         self._combine = [(d, l) for d, l in zip(self._data, self._label)]

#         self._sampler = Sampler(datas=self._combine)

#     def visualize(self, image, **kwargs):
#         if self._is_norm and not self._is_bernolli:
#             image += 0.5
#         output = image.reshape([-1, 28, 28])
#         if self._is_batch:
#             output = list(output)
#         return output

#     def _sample_data_label_weight(self):
#         sample = next(self._sampler)
#         sample = sample[0]
#         image = sample[0]
#         digit = sample[1]
#         if self._is_bin:
#             image[image < 125.1] = 0.0
#             image[image >= 125.0] = 255.0
#         if self._is_norm:
#             image = image / 256.0
#             if not self._is_bernolli:
#                 image -= 0.5
#         image = image.reshape([np.prod(image.shape)])

#         if self._is_unsp:
#             label = np.copy(image)
#         else:
#             label = digit
#         if self._is_noise:
#             if self._noise_type == 'poisson':
#                 image = np.random.poisson(
#                     image / self._noise_scale).astype(np.float32)
#                 image *= (self._noise_scale / 2.0)
#             elif self._noise_type == "gaussian":
#                 args = image.shape
#                 noise = np.random.randn(*args) * self._noise_scale
#                 image = image + noise
#         return (image, label, 1.0)


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


class MNIST2(DataSetImages):

    @with_config
    def __init__(self,
                 is_flatten=False,
                 is_only_label=False,
                 settings=None,
                 **kwargs):
        DataSetBase.__init__(self, **kwargs)
        self._settings = settings
        self._is_flatten = self._update_settings('is_flatten', is_flatten)
        self._is_only_label = self._update_settings(
            'is_only_label', is_only_label)
        self._fin = h5py.File(os.path.join(PATH_DATASETS, 'mnist2.h5'), 'r')
        if self._is_train:
            self._dataset = self._fin['train']
        else:
            self._dataset = self._fin['test']
        self._images = self._dataset['image']
        self._labels = self._dataset['label']
        self._nb_examples = self._images.shape[0]
        ids = list(range(self._nb_examples))
        self._sampler = Sampler(ids, is_shuffle=self._is_train)
        if self._is_only_label:
            self._nb_data = 1
        else:
            self._nb_data = 2

    def visualize(self, image, data_type='data'):
        image = np.float32(image)
        if self._is_flatten:
            if data_type == 'data':
                output = image.reshape([-1, 28 * 2, 28 * 2])
            else:
                output = image.reshape([-1, 28, 28])
        else:
            output = image
        if self._is_batch:
            output = list(output)
        return output

    def _sample_data_label_weight(self):
        id_ = next(self._sampler)
        data = np.array(self._images[id_[0], :, :, :], dtype=np.float16)
        label = np.array(self._labels[id_[0], :, :, :], dtype=np.float16)
        if self._is_flatten:
            data = data.reshape((-1,))
            label = label.reshape((-1,))
        if self._is_only_label:
            return (label, data, 1.0)
        else:
            return ((data, label), data, 1.0)


# def write_dataset_to_tfrecords():
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#     nb_train = x_train.shape[0]
#     nb_test = x_test.shape[0]
#     with tf.python_io.TFRecordWriter("MNIST.train.tfrecords") as writer:
#         for i in tqdm(range(nb_train)):
#             bytes = np.reshape(x_train[i, ...], (-1,)
#                                ).astype(np.uint8).tobytes()
#             bytes_f = tf.train.Feature(
#                 bytes_list=tf.train.BytesList(value=[bytes]))
#             shape_f = tf.train.Feature(
#                 int64_list=tf.train.Int64List(value=[28, 28]))
#             label_f = tf.train.Feature(
#                 int64_list=tf.train.Int64List(value=[y_train[i]]))
#             example = tf.train.Example(features=tf.train.Features(feature={
#                 'shape': shape_f,
#                 'image': bytes_f,
#                 'label': label_f
#             }))
#             writer.write(example.SerializeToString())

#     with tf.python_io.TFRecordWriter("MNIST.test.tfrecords") as writer:
#         for i in tqdm(range(nb_test)):
#             bytes = np.reshape(x_test[i, ...], (-1,)
#                                ).astype(np.uint8).tobytes()
#             bytes_f = tf.train.Feature(
#                 bytes_list=tf.train.BytesList(value=[bytes]))
#             shape_f = tf.train.Feature(
#                 int64_list=tf.train.Int64List(value=[28, 28]))
#             label_f = tf.train.Feature(
#                 int64_list=tf.train.Int64List(value=[y_test[i]]))
#             example = tf.train.Example(features=tf.train.Features(feature={
#                 'shape': shape_f,
#                 'image': bytes_f,
#                 'label': label_f
#             }))
#             writer.write(example.SerializeToString())


# def check_load():
#     record_iterator = tf.python_io.tf_record_iterator(
#         path='MNIST.test.tfrecords')
#     imgs = []
#     label = []
#     for i in range(5):
#         string_record = next(record_iterator)
#         example = tf.train.Example()
#         example.ParseFromString(string_record)
#         imgs.append(np.fromstring(example.features.feature[
#                     'image'].bytes_list.value[0], dtype=np.uint8))
#         label.append(int(example.features.feature[
#                      'label'].int64_list.value[0]))
#     return imgs, label

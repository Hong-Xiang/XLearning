""" base class for samples generator for Keras models."""
# TODO: batch mode ?
# TODO: Add tensorflow queue support?
import os
import json
import random
import logging
import numpy
import h5py

from ..utils.general import with_config, empty_list
from ..utils.tensor import down_sample_nd
from ..utils.cells import Sampler

PATH_DATASETS = os.environ['PATH_DATASETS']
DDTYPE = numpy.float16


class DataSetBase(object):
    """ base class of general dataset

        # Initialization
            self._initialize()
            self.__enter__()

        # Finalize
            self.__exit__()
    """
    @with_config
    def __init__(self,
                 is_batch=True,
                 batch_size=32,
                 is_train=True,
                 is_label=True,
                 is_unsp=False,
                 is_weight=False,
                 random_seed=None,
                 is_norm=False,
                 norm_c=None,
                 is_noise=False,
                 noise_level=0.0,
                 noise_type='gaussian',
                 is_finite=True,
                 file_data=None,
                 dataset_name='images',
                 settings=None,
                 **kwargs):
        self._c = dict()
        self._settings = settings

        self._is_batch = self._update_settings('is_batch', is_batch)
        self._batch_size = self._update_settings('batch_size', batch_size)

        self._is_train = self._update_settings('is_train', is_train)

        self._is_label = self._update_settings('is_label', is_label)
        self._is_weight = self._update_settings('is_weight', is_weight)

        # is_unsp: is unsupervised learning, if is_unsp is True,
        # self.sample_data_label will return (data, data)
        self._is_unsp = self._update_settings('is_unsp', is_unsp)

        self._random_seed = self._update_settings('random_seed', random_seed)

        self._is_norm = self._update_settings('is_norm', is_norm)
        self._norm_c = self._update_settings('norm_c', norm_c)

        self._is_noise = self._update_settings('is_noise', is_noise)
        self._noise_level = self._update_settings('noise_level', noise_level)
        self._noise_type = self._update_settings('noise_type', noise_type)

        # Read data from file or by generate
        self._is_filedata = False
        self._file_data = self._update_settings('file_data', file_data)
        self._dataset_name = self._update_settings(
            'dataset_name', dataset_name)
        self._is_finite = self._update_settings('is_finite', is_finite)

        # Total number of examples (only valid when self._is_finite = True)
        self._nb_examples = 0

        self._is_init = False

        if self._random_seed is not None:
            numpy.random.seed(self._random_seed)
            random.seed(self._random_seed)

        self._nb_data = 1
        self._nb_label = 1

    def _update_settings(self, name, value=None):
        output = self._settings.get(name, value)
        self._c.update({name: output})
        return output

    def randint(self, minv=0, maxv=100):
        return random.randint(minv, maxv)

    def randnorm(self, mean=0, sigma=1, size=None):
        if size is None:
            s = numpy.random.normal(loc=mean, scale=sigma)
        else:
            s = numpy.random.normal(loc=mean, scale=sigma, size=size)
        return s

    def randunif(self, minv=0.0, maxv=1.0, size=None):
        if size is None:
            s = numpy.random.uniform(minv, maxv)
        else:
            s = numpy.random.uniform(minv, maxv, size)
        return s

    def _initialize(self):
        self._is_init = True

    def __enter__(self):
        self._is_init = True
        if self._is_filedata:
            self._fin = h5py.File(self._file_data, 'r')
            if self._is_train:
                self._dataset = self._fin.get('train')
            else:
                self._dataset = self._fin.get('test')
            if self._dataset is None:
                self._dataset = self._fin
            self._dataset = self._dataset[self._dataset_name]
            self._nb_examples = self._dataset.shape[0]
            if self._is_train:
                self._sampler = Sampler(datas=range(
                    self._nb_examples), is_shuffle=True)
            else:
                self._sampler = Sampler(datas=range(
                    self._nb_examples), is_shuffle=False)
        return self

    def __exit__(self, etype, value, traceback):
        self._fin.close()

    def _sample_data(self):
        """ Genreate a new sample, data only """
        sample = self._sample_data_label()
        return (sample[0],)

    def _sample_data_label(self):
        """ Generate a new sample, (data, label) """
        sample = self._sample_data_label_weight()
        if self._is_unsp:
            sample = (sample[0], sample[0])
        return (sample[0], sample[1])

    def _sample_data_label_weight(self):
        """ Generate a new sample, (data, label, weight) """
        raise NotImplementedError("sample_data_label_weight not implemented.")

    def pretty_settings(self):
        """ return settings in pretty .json string """
        return json.dumps(self._c, sort_keys=True, indent=4, separators=(',', ':'))

    def visualize(self, sample):
        """ Convert sample into visualizeable format """
        raise NotImplementedError("Visualize is not implemented.")

    def data_from_sample(self, sample, data_type='data'):
        """ Get data from sample """
        if data_type == 'weight':
            if self._is_weight:
                return sample[2]
            else:
                raise TypeError("No weight in sample.")
        if data_type == 'label':
            if len(sample) > 1:
                return sample[1]
            else:
                raise TypeError("No label in sample.")
        if data_type == 'data':
            return sample[0]
        raise ValueError("Unknown data_type {0}.".format(data_type))

    def _sample_single(self):
        """ interface of sampling """
        if self._is_weight:
            return self._sample_data_label_weight()
        elif self._is_label:
            return self._sample_data_label()
        else:
            return self._sample_data()

    def _sample_batch(self):
        """ form mini batch """
        all_example = []
        for i in range(self._batch_size):
            all_example.append(self._sample_data_label_weight())
        samples = []
        x = empty_list(self._nb_data)
        for j in range(self._nb_data):
            x[j] = []
            for i in range(self._batch_size):
                x[j].append(all_example[i][0][j])
            x[j] = numpy.array(x[j], dtype=DDTYPE)
        samples.append(x)
        if self._is_label:
            y = empty_list(self._nb_label)
            for j in range(self._nb_label):
                y[j] = []
                for i in range(self._batch_size):
                    y[j].append(all_example[i][1][j])
                y[j] = numpy.array(y[j], dtype=DDTYPE)
            samples.append(y)
        if self._is_weight:
            w = []
            for i in range(self._batch_size):
                w.append(all_example[i][2])
            w = numpy.array(w)
            samples.append(w)
        return samples

    def __next__(self):
        if not self._is_init:
            self._initialize()
        if self._is_batch:
            return self._sample_batch()
        else:
            return self._sample_single()

    def gather_examples(self, nb_examples=64):
        """ gather given numbers of examples """
        nb_gathered = 0
        s = None
        is_weight = None
        while nb_gathered < nb_examples:
            tmp = self.__next__()
            if s is None:
                s = tmp
                is_weight = len(s) > 1
            else:
                nb_gathered += tmp[0][0].shape[0]
                for i, e in enumerate(tmp[0]):
                    s[0][i] = numpy.concatenate((s[0][i], tmp[0][i]))
                if is_weight:
                    for i, e in enumerate(tmp[1]):
                        s[1][i] = numpy.concatenate((s[1][i], tmp[1][i]))
        return s

    @property
    def batch_size(self):
        return self._batch_size

# TODO: impl is_crop_center


class DataSetImages(DataSetBase):
    """ base class for image datasets """
    @with_config
    def __init__(self,
                 is_gray=True,
                 is_uint8=True,
                 is_crop=False,
                 is_crop_center=False,
                 crop_target_shape=None,
                 crop_offset=(0, 0),
                 is_crop_random=True,
                 is_down_sample=False,
                 nb_down_sample=3,
                 is_down_sample_0=True,
                 is_down_sample_1=True,
                 down_sample_method='mean',
                 settings=None,
                 **kwargs):
        super(DataSetImages, self).__init__(**kwargs)
        self._settings = settings
        self._is_gray = self._update_settings('is_gray', is_gray)
        self._is_uint8 = self._update_settings('is_uint8', is_uint8)
        self._is_down_sample = self._update_settings(
            'is_down_sample', is_down_sample)
        self._is_down_sample_0 = self._update_settings(
            'is_down_sample_0', is_down_sample_0)
        self._is_down_sample_1 = self._update_settings(
            'is_down_sample_1', is_down_sample_1)
        self._nb_down_sample = self._update_settings(
            'nb_down_sample', nb_down_sample)
        self._down_sample_method = self._update_settings(
            'down_sample_method', down_sample_method)
        self._is_crop = self._update_settings('is_crop', is_crop)
        self._is_crop_center = self._update_settings(
            'is_crop_center', is_crop_center)
        self._crop_target_shape = self._update_settings(
            'crop_target_shape', crop_target_shape)
        self._crop_offset = self._update_settings('crop_offset', crop_offset)
        self._is_crop_random = self._update_settings(
            'is_crop_random', is_crop_random)
        self._down_sample_ratio = [1, 1]
        if self._is_down_sample_0:
            self._down_sample_ratio[0] = 2
        if self._is_down_sample_1:
            self._down_sample_ratio[1] = 2

        self._is_finite = True

        self._fin = None
        self._sampler = None
        self._dataset = None

        if self._is_uint8:
            self._norm_c = 256.0
        else:
            self._norm_c = 1.0

        if self._is_down_sample:
            self._nb_data = self._nb_down_sample + 1

    def _crop(self, image):
        """ crop image into small patch """
        image = numpy.array(image, dtype=image.dtype)
        target_shape = self._crop_target_shape
        offsets = list(self._crop_offset)
        is_crop_random = self._is_crop_random
        if is_crop_random:
            offsets[0] += self.randint(minv=0,
                                       maxv=image.shape[0] - target_shape[0])
            offsets[1] += self.randint(minv=0,
                                       maxv=image.shape[1] - target_shape[1])
        if len(image.shape) == 3:
            image = image[offsets[0]:offsets[0] +
                          target_shape[0], offsets[1]:offsets[1] + target_shape[1], :]
        else:
            image = image[offsets[0]:offsets[0] +
                          target_shape[0], offsets[1]:offsets[1] + target_shape[1]]
        return image

    def _downsample(self, image):
        """ down sample image/patch """
        image = numpy.array(image, dtype=image.dtype)
        image_d = down_sample_nd(image, list(
            self._down_sample_ratio) + [1], method=self._down_sample_method)
        return image_d

    def visualize(self, sample):
        """ convert numpy.ndarray data into list of plotable images """
        # Decouple visualization of data

        images = numpy.array(sample, dtype=numpy.float32)

        # Remove last axis for gray images.
        if sample.shape[-1] == 1:
            images = images.reshape(images.shape[:-1])
        else:
            images = images

        # Inverse normalization
        if self._is_norm:
            images += 0.5
            images *= self._norm_c

        if self._is_uint8:
            images = numpy.array(images, dtype=numpy.uint8)
        # Divide batched numpy.ndarray into list of images
        if self._is_batch:
            images = list(images)
        return images

    def _sample_data_label_weight(self):
        """ read from dataset HDF5 file and perform necessary preprocessings """
        idx = next(self._sampler)[0]
        image = numpy.array(self._dataset[idx], dtype=DDTYPE)
        if self._is_crop:
            # read next example until image large enough
            failed = 0
            min_shape = (self._crop_offset[0] + self._crop_target_shape[0],
                         self._crop_offset[1] + self._crop_target_shape[1])
            while image.shape[0] < min_shape[0] or image.shape[1] < min_shape[1]:
                idx = next(self._sampler)[0]
                image = numpy.array(self._dataset[idx], dtype=DDTYPE)
                failed += 1
                if failed > 100:
                    raise ValueError('Failed to get proper sized images with crop_offset: {0}, crop_target_shape {1}, image_shape{2}.'.format(
                        self._crop_offset, self._crop_target_shape, image.shape))
            # crop sample
            image = self._crop(image)

        if self._is_gray:
            image = numpy.mean(image, axis=-1, keepdims=True)

        if self._is_norm:
            image /= self._norm_c
            image -= 0.5

        # Down sample
        if self._is_down_sample:
            label = numpy.array(image, dtype=DDTYPE)
            imgs = []
            imgs.append(numpy.array(image, dtype=DDTYPE))
            for i in range(self._nb_down_sample):
                imgs.append(numpy.array(
                    self._downsample(imgs[i]), dtype=DDTYPE))
        else:
            label = image
        if self._is_down_sample:
            sample = (imgs, (label,), 1.0)
        else:
            sample = ((image,), (label,), 1.0)
        return sample

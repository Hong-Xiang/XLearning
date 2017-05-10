""" base class for samples generator for Keras models."""
# TODO: batch mode ?
# TODO: Add tensorflow queue support?
# TODO: Put options into configs
import os
import json
import random
import numpy
import h5py
import pathlib

from ..utils.prints import pp_json
from ..utils.general import with_config, empty_list
from ..utils.tensor import down_sample_nd
from ..utils.cells import Sampler

PATH_DATASETS = os.environ['PATH_DATASETS']

class DataSetBase(object):
    """ base class of general dataset
        # Sample
        ## methods:
            sample()
            __next__()
        ##
            samper from a dataset.



        # Initialization
            self._initialize()
            self.__enter__()

        # Finalize
            self.__exit__()
    """
    @with_config
    def __init__(self,
                 batch_size=32,
                 mode='train',
                 random_seed=None,
                 is_norm=True,
                 data_mean=0.0,
                 data_std=1.0,
                 file_data=None,
                 name='dataset',
                 **kwargs):
        """
        Dataset Base class
        """
        self.batch_size = batch_size
        self.mode = mode
        self.random_seed = random_seed
        self.data_mean = data_mean
        self.data_std = data_std
        self.file_data = file_data
        self.name = name
        self.is_norm = is_norm
        self.keys = ['data', 'label']
        self.pp_dict = {
            'batch_size': self.batch_size,
            'mode': self.mode,
            'random_seed': self.random_seed,
            'data_mean': self.data_mean,
            'data_std': self.data_std,
            'file_data': self.file_data,
            'is_norm': self.is_norm,
            'keys': self.keys
        }

        self.nb_examples = None
        self.sampler = None

    def rand_int(self, minv=0, maxv=100):
        return random.randint(minv, maxv)

    def rand_norm(self, mean=0, sigma=1, size=None):
        s = numpy.random.normal(loc=mean, scale=sigma, size=size)
        return s

    def rand_unif(self, minv=0.0, maxv=1.0, size=None):
        s = numpy.random.uniform(low=minv, high=maxv, size=size)
        return s

    def initialize(self):
        self._is_init = True
        self.print_settings()

    def __enter__(self):
        self.initialize()
        return self

    def finalize(self):
        pass

    def __exit__(self, etype, value, traceback):
        self.finalize()

    def sample(self):
        """ Genreate a new sample
        Returns:
            A dict of mini-batch tensors.
        """
        out = {}
        for k in self.keys:
            out[k] = []
        for i in range(self.batch_size):
            ss = self._sample_single()
            for k in self.keys:
                out[k].append(ss[k])
        for k in self.keys:
            out[k] = numpy.array(out[k])
        return out

    def norm(self, ip_):
        if self.is_norm:
            normed = ip_ - self.data_mean
            normed = normed / self.data_std
        else:
            return ip_
        return normed

    def denorm(self, ip_):
        if self.is_norm:
            denormed = ip_ * self.data_std
            denormed = denormed + self.data_mean
        else:
            return ip_
        return denormed

    def print_settings(self):
        pp_json(self.pp_dict, title="DATASET SETTINGS:")

    def visualize(self, sample):
        """ Convert sample into visualizeable format """
        raise NotImplementedError

    def _sample_single(self):
        """ interface of sampling """
        raise NotImplementedError

    def __next__(self):
        if not self._is_init:
            raise ValueError('Not initilized.')
        return self.sample()

    def gather_examples(self, nb_examples=64):
        """ gather given numbers of examples """
        nb_samples = int(numpy.ceil(nb_examples / self.batch_size))
        out = None
        for _ in range(nb_samples):
            s = self.sample()
            if out is None:
                out = s
            else:
                for k in self.keys:
                    out[k] = numpy.concatenate([out[k], s[k]], axis=0)
        for k in self.keys:
            out[k] = out[k][:nb_examples, ...]
        return out


class DataSetImages(DataSetBase):
    """ base class for image datasets
        All images in NCHW format, i.e. channel_first
    """
    @with_config
    def __init__(self,
                 is_uint8=True,
                 is_gray=True,
                 crop_shape=None,
                 crop_offset=(0, 0),
                 is_crop_random=True,
                 is_down_sample=False,
                 data_down_sample=3,
                 label_down_sample=0,
                 is_down_sample_0=True,
                 is_down_sample_1=True,
                 down_sample_method='mean',
                 data_key='images',
                 is_from_npy=False,
                 nnz_ratio=0.0,
                 padding=None,
                 period=None,
                 data_format='channels_last',
                 **kwargs):
        super(DataSetImages, self).__init__(**kwargs)
        self.is_gray = is_gray
        self.is_uint8 = is_uint8
        self.is_down_sample = is_down_sample
        self.is_down_sample_0 = is_down_sample_0
        self.is_down_sample_1 = is_down_sample_1
        self.data_down_sample = data_down_sample
        self.label_down_sample = label_down_sample
        self.down_sample_method = down_sample_method
        self.crop_shape = crop_shape
        self.crop_offset = crop_offset
        self.is_crop_random = is_crop_random
        self.down_sample_ratio = [1, 1]
        if self.is_down_sample_0:
            self.down_sample_ratio[0] = 2
        if self.is_down_sample_1:
            self.down_sample_ratio[1] = 2
        self.data_key = data_key
        self.is_from_npy = is_from_npy
        self.nnz_ratio = nnz_ratio
        self.padding = padding
        self.period = period
        self.data_format = data_format
        self.pp_dict.update({
            'padding': self.padding,
            'period': self.period,
            'is_gray': self.is_gray,
            'is_uint8': self.is_uint8,
            'is_down_sample': self.is_down_sample,
            'is_down_sample_0': self.is_down_sample_0,
            'is_down_sample_1': self.is_down_sample_1,
            'data_down_sample': self.data_down_sample,
            'label_down_sample': self.label_down_sample,
            'down_sample_method': self.down_sample_method,
            'crop_shape': self.crop_shape,
            'crop_offset': self.crop_offset,
            'is_crop_random': self.is_crop_random,
            'down_sample_ratio': self.down_sample_ratio,
            'data_key': self.data_key,
            'is_from_npy': self.is_from_npy,
            'nnz_ratio': self.nnz_ratio,
            'data_format': self.data_format
        })
        self._sampler = None
        self._dataset = None
        self._fin = None

    def crop(self, image_ip):
        """ crop image into small patch """
        image = numpy.array(image_ip, dtype=numpy.float32)
        target_shape = self.crop_shape
        offsets = list(self.crop_offset)
        is_crop_random = self.is_crop_random
        if is_crop_random:
            max0 = image.shape[1] - target_shape[0] - 1
            if max0 > 0:
                offsets[0] += self.randint(minv=0, maxv=max0)
            else:
                offsets[0] = 0
            max1 = image.shape[2] - target_shape[1] - 1
            if max1 > 0:
                offsets[1] += self.randint(minv=0, maxv=max1)
            else:
                offsets[1] = 0
        if offsets[0] + target_shape[0] > image.shape[1]:
            raise ValueError('Too large crop shape or offset.')
        if offsets[1] + target_shape[1] > image.shape[2]:
            raise ValueError('Too large crop shape or offset.')

        image = image[:,
                      offsets[0]:offsets[0] + target_shape[0],
                      offsets[1]:offsets[1] + target_shape[1]]

        return image

    def downsample(self, image):
        """ down sample image/patch """
        image = numpy.array(image, dtype=numpy.float32)

        image_d = down_sample_nd(image, [1] + list(
            self.down_sample_ratio), method=self.down_sample_method)

        return image_d

    def visualize(self, sample, is_no_change=False):
        """ convert numpy.ndarray data into list of plotable images """
        # Decouple visualization of data
        images = numpy.array(sample, dtype=numpy.float32)

        # Remove axis 1 for gray images.
        if sample.shape[1] == 1:
            images = images.reshape(
                [images.shape[0], images.shape[2], images.shape[3]])
        else:
            images = images

        # Inverse normalization
        if not is_no_change:
            images = self.denorm(images)

        if self.is_uint8 and not is_no_change:
            images = numpy.array(images, dtype=numpy.uint8)
        images = list(images)
        return images

    def initialize(self):
        super(DataSetImages, self).initialize()
        p = pathlib.Path(self.file_data)
        if self.is_from_npy:
            self.fin = str(p.absolute())
            self._dataset = numpy.load(self.fin)
        else:
            self.fin = h5py.File(p, 'r')
            self._dataset = self.fin[self.data_key]
        if self.is_from_npy:
            nb_data = self._dataset.shape[0]
            nb_train = nb_data // 5 * 4
            if self.mode == 'train':
                self.nb_examples = nb_train
                self._sampler = Sampler(
                    list(range(self.nb_examples)), is_shuffle=True)
            else:
                self.nb_examples = nb_data - nb_train
                self._sampler = Sampler(
                    list(range(nb_train, nb_data)), is_shuffle=True)
        else:
            self.nb_examples = self._dataset.shape[0]
            self._sampler = Sampler(
                list(range(self.nb_examples)), is_shuffle=False)

    def finalize(self):
        super(DataSetImages, self).finalize()
        if not self.is_from_npy:
            self.fin.close()

    def _load_sample(self):
        failed = True
        nb_failed = 0
        while failed:
            failed = False
            idx = next(self._sampler)[0]
            image = numpy.array(self._dataset[idx], dtype=numpy.float32)
            if len(image.shape) == 2:
                image = image.reshape([1, image.shape[0], image.shape[1]])
            elif image.shape[2] == 1:
                image = numpy.transpose(image, [2, 0, 1])
            if self.padding is not None:
                image = image[:, :self.period[0], :self.period[1]]
                images = [image] * (self.padding[0] + 1)
                image = numpy.concatenate(images, axis=1)
                images = [image] * (self.padding[1] + 1)
                image = numpy.concatenate(images, axis=2)
            if image.shape[1] < self.crop_offset[0] + self.crop_shape[0]:
                failed = True
            if image.shape[2] < self.crop_offset[1] + self.crop_shape[1]:
                failed = True
            if not failed:
                image = self.crop(image)
            nnz = len(numpy.nonzero(image > 1e-5)[0])
            rnz = nnz / numpy.size(image)
            if rnz < self.nnz_ratio:
                failed = True
            nb_failed += 1
            if nb_failed > 100:
                raise ValueError('Tried load more than 100 images and failed.')
        return image

    def _sample_single(self):
        """ read from dataset HDF5 file and perform necessary preprocessings """
        image = self._load_sample()

        if self.is_gray:
            image = numpy.mean(image, axis=0, keepdims=True)

        if self.is_norm:
            image = self.norm(image)

        return {'data': image, 'label': None}

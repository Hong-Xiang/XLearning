""" base class for samples generator for Keras models."""
# TODO: batch mode ?

import random
import numpy
import logging
import xlearn.utils.general as utg


class DataSetBase(object):
    """ Base class of dataset """
    @utg.with_config
    def __init__(self,
                 is_batch=False,
                 batch_size=128,
                 is_label=True,
                 is_weight=False,
                 random_seed=None,
                 is_norm=False,
                 is_train=True,
                 is_noise=False,
                 noise_level=0.0,
                 noise_type='gaussian',
                 filenames=None,
                 is_finite=True,
                 is_para_sample=False,
                 settings=None,
                 **kwargs):
        self._settings = settings
        logging.getLogger(__name__).debug(self._settings)
        self._is_label = self._settings.get('is_label', is_label)
        self._is_weight = self._settings.get('is_weight', is_weight)
        self._is_batch = self._settings.get('is_batch', is_batch)
        self._batch_size = self._settings.get('batch_size', batch_size)
        self._random_seed = self._settings.get('random_seed', random_seed)
        if self._random_seed is not None:
            numpy.random.seed(self._random_seed)
            random.seed(self._random_seed)
        self._is_norm = self._settings.get('is_norm', is_norm)
        self._is_train = self._settings.get('is_train', is_train)
        self._is_noise = self._settings.get('is_noise', is_noise)
        self._noise_level = self._settings.get('noise_level', noise_level)
        self._noise_type = self._settings.get('noise_type', noise_type)

        self._is_finite = is_finite
        self._nb_datas = 0
        self._is_para_sample = is_para_sample

        self._is_init = False

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
        pass

    def __exit__(self, type, value, traceback):
        pass

    def _sample_data(self):
        """ Genreate a new sample, data only """
        sample = self._sample_data_label()
        return (sample[0],)

    def _sample_data_label(self):
        """ Generate a new sample, (data, label) """
        sample = self._sample_data_label_weight()
        return (sample[0], sample[1])

    def _sample_data_label_weight(self):
        """ Generate a new sample, (data, label, weight) """
        raise TypeError("sample_data_label_weight not implemented.")

    def visualize(self, sample, **kwargs):
        raise TypeError("Visualize is not implemented.")

    def data_from_sample(self, sample, data_type='data'):
        if data_type == 'weight':
            if self._is_weight:
                return sample[2]
            else:
                raise TypeError("No wieght in sample.")
        if data_type == 'label':
            if len(sample) > 1:
                return sample[1]
            else:
                raise TypeError("No label in sample.")
        if data_type == 'data':
            return sample[0]
        raise ValueError("Unknown data_type.")

    def _sample_single(self):
        if self._is_label:
            if self._is_weight:
                return self._sample_data_label_weight()
            else:
                return self._sample_data_label()
        else:
            return self._sample_data()

    def _sample_batch(self):
        if not self._is_batch:
            raise ValueError(
                "Can not call _sample_batch when _is_batch == False.")
        if self._is_label:
            if self._is_weight:
                x = []
                y = []
                w = []
                for i in range(self._batch_size):
                    s = self._sample_data_label_weight()
                    x.append(s[0])
                    y.append(s[1])
                    w.append(s[2])
                x = numpy.array(x)
                y = numpy.array(y)
                w = numpy.array(w)
                samples = (x, y, w)
            else:
                x = []
                y = []
                for i in range(self._batch_size):
                    s = self._sample_data_label()
                    x.append(s[0])
                    y.append(s[1])
                x = numpy.array(x)
                y = numpy.array(y)
                samples = (x, y)
        else:
            x = []
            for i in range(self._batch_size):
                s = self._sample_data()
                x.append(s[0])
            x = numpy.array(x)
            samples = (x, )
        return samples

    def __next__(self):
        if not self._is_init:
            self._initialize()
        if self._is_batch:
            return self._sample_batch()
        else:
            return self._sample_single()


class DataSetImages(DataSetBase):

    def __init__(self, datafile=None, is_4d=True, is_gray=True, is_down_sample=False, down_sample_ratio=None, **kwargs):
        super(DataSetImages, self).__init__(
            datafile=datafile, is_4d=is_4d, is_gray=is_gray, **kwargs)
        self._fn_data = self._settings['fn_data']
        self._is_4d = self._settings['is_4d']
        self._is_gray = self._settings['is_gray']
        self._is_down_sample = self._settings['is_down_sample']
        self._down_sample_ratio = self._settings['down_sample_ratio']

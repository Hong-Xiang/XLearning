""" base class for samples generator for Keras models."""
# TODO: batch mode ?

import random
import numpy
import logging
import xlearn.utils.general as utg


class DataSetBase(object):
    """ Base class of dataset """
    @utg.with_config
    def __init__(self, is_batch=False, batch_size=128, is_label=True, is_weight=False, random_seed=None, filenames=None, is_finite=True, settings=None, **kwargs):
        self._settings = settings
        logging.getLogger(__name__).debug(self._settings)
        self._is_label = is_label
        self._is_weight = is_weight
        self._is_batch = is_batch
        self._batch_size = batch_size
        if random_seed is not None:
            numpy.random.seed(random_seed)
            random.seed(random_seed)
        self._is_init = False

        self._is_finite = is_finite

    def _initialize(self):
        self._is_init = True

    # def _nb_samples(self):
    #     return 0

    def _sample_data(self):
        """ Genreate a new sample, data only """
        sample = self._sample_data_label()
        return sample[0]

    def _sample_data_label(self):
        """ Generate a new sample, (data, label) """
        sample = self._sample_data_label_weight()
        return (sample[0], sample[1])

    def _sample_data_label_weight(self):
        """ Generate a new sample, (data, label, weight) """
        raise TypeError("sample_data_label_weight not implemented.")

    def visualize(self, sample, **kwargs):
        raise TypeError("Visualize is not implemented.")

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

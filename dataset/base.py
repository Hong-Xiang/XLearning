""" base class for samples generator for Keras models."""
# TODO: batch mode ?

import random
import numpy

class DataSetBase(object):
    """ Base class of dataset """

    def __init__(self, is_batch=False, is_label=True, is_weight=False, random_seed=None):
        self.is_label = is_label
        self.is_weight = is_weight
        self.is_batch = is_batch
        if random_seed is not None:
            numpy.random.seed(random_seed)
            random.seed(random_seed)

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

    def __next__(self):
        if self.is_label:
            if self.is_weight:
                return self._sample_data_label_weight()
            else:
                return self._sample_data_label()
        else:
            return self._sample_data()

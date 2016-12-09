"""base class of datasets
"""
from abc import ABCMeta, abstractmethod
import json
import numpy as np
import xlearn.utils.general as utg
import xlearn.utils.xpipes as utp


class DataSet(metaclass=ABCMeta):
    """Base class of all dataset classes.
    """

    def __init__(self, filenames=None, **kwargs):
        super(DataSet, self).__init__()
        self._paras = utg.merge_settings(filenames=filenames, **kwargs)
        self._dataset_type = self._paras['dataset_type']
        self._gather_paras("common")
        self._gather_paras(self._dataset_type)

        self._prepare()

    def _gather_paras(self, dataset_type):
        """Gather parameters from self._paras to shortcuts like self._batch_size.
        """
        if dataset_type == "common":
            self._batch_size = self._paras['batch_size']
            self._shape_i = self._paras['shape_i']
            if self._shape_i[0] != 1:
                raise ValueError(utg.errmsg(
                    self._shape_i[0], 1, msg="Invalid first dimension of shape_i, "))
            self._shape_o = self._paras['shape_o']
            if self._shape_o[0] != 1:
                raise ValueError(utg.errmsg(
                    self._shape_o[0], 1, msg="Invalid first dimension of shape_i, "))
            self._is_pad = self._paras['is_pad']
            self._is_cache = self._paras['is_cache']
            self._epoch_max = self._paras['epoch_max']

            self._is_shuffle = None
            if 'is_shuffle' in self._paras:
                self._is_shuffle = self._paras['is_shuffle']
            if self._is_shuffle is None:
                self._is_shuffle = self._is_train_or_test

    def _prepare(self):
        """Prepare work before dataset is ready to use.
        """
        self._epoch = utp.Counter(max_state=self._epoch_max)

    def _next_epoch(self):
        next(self._epoch.out)

    def _batch_start(self):
        """Called at start of next_batch()
        """
        pass

    def _batch_end(self):
        """Called at end of next_batch()
        """
        pass

    def next_batch(self):
        """Returns next batch of current dataset.
        """

        self._batch_start()

        if self._is_train_or_test or self._is_cache:
            tensor_data = np.zeros(self._shape_i)
            tensor_label = np.zeros(self._shape_o)
            for i in range(self.batch_size):
                try:
                    data, label = self._single_sample()
                except StopIteration:
                    data, label = self._triger_stop_iteration()
                if data is None:
                    tensor_data = tensor_data[:i]
                    tensor_label = tensor_label[:i]
                    break
                else:
                    tensor_data[i, :] = data
                    tensor_label[i, :] = label
        else:
            tensor_data = np.zeros(self._shape_i)
            for i in range(self.batch_size):
                try:
                    data = self._single_sample()
                except StopIteration:
                    data = self._triger_stop_iteration()
                if data is None:
                    tensor_data = tensor_data[:i]
                    break
                else:
                    tensor_data[i, :] = data

        self._batch_end()

        if self._is_train_or_test:
            return tensor_data, tensor_label
        else:
            return tensor_data

    def _is_train_or_test(self):
        return self._dataset_type == "train" or self._dataset_type == "test"

    def _triger_stop_iteration(self):
        """called when self._single_sample() raise a StopIteration Exception.
        """
        return self._padding()

    def _padding(self):
        """padding batch data when used up all datas and a batch is not finished
        """
        if not self._is_pad:
            if self._is_train_or_test:
                return None, None
            else:
                return None
        else:
            return self._single_sample()

    @abstractmethod
    def _single_sample(self):
        """Returns a tensor of shape [1, *] or tuple ([1, *], [1, *])
        """
        if self._is_train_or_test:
            return np.zeros(self._shape_i), np.zeros(self._shape_o)
        else:
            return np.zeros(self._shape_i)

    def print_paras(self):
        """Print all config parameters.
        """
        print(self._paras)

    @property
    def batch_size(self):
        """Batch size"""
        return self._batch_size

    @property
    def dataset_type(self):
        """Dataset type, one of following:
            - train
            - test
            - infer
        """
        return self._dataset_type

    @property
    def epoch(self):
        """# of finished epoches"""
        return self._epoch.state

    @property
    def epoch_max(self):
        """Maximum epoch"""
        return self._epoch.max_state

    @property
    def is_shuffle(self):
        """is shuffle"""
        return self._is_shuffle

    @property
    def is_pad(self):
        """is pad to fill batch"""
        return self._is_pad

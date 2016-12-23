"""base class of datasets
"""
import logging
import numpy as np
import xlearn.utils.general as utg
import xlearn.utils.xpipes as utp


class EndEpoch(StopIteration):
    pass


class EndSingleFile(StopIteration):
    pass


class NoMoreEpoch(StopIteration):
    pass


class DataSet(object):
    """Base class of all dataset classes.
    """

    # def __init__(self, filenames=None, **kwargs):
    def __init__(self, filenames=None, **kwargs):
        """Initialization of DataSet class.

        This base class handles reading parameters, epoch updates and batch
        generation. Inherence class should implement following method:
        Args:
            filenames: A list of str, configure .JSON files.
            kwargs: directly input settings.

        Optional:
            _set_paras_default(self): set default paras, for self._paras only

            _gather_paras_common(self): for all dataset types
            _gather_paras_train(self):
            _gather_paras_test(self):
            _gather_paras_infer(self):
            _gather_paras_special(self): other paras

            _prepare(self): called after all paras getted, before batch generation

        Direct refernce to self._para outside self._gather_paras(conf_type)
        is *NOT* allowed.

        Batch generation:
        Required:
            _sample(self)
                Returns: [1, *], [1, *] tensors
        Optional:
            _padding(self)
            _end_of_epoch(self):
                By default, it will return self._padding()
                Returns: [1, *], [1, *] tensors

            _batch_start(self)
            _batch_end(self)
        """
        self._paras = utg.merge_settings(
            filenames=filenames, default_settings=self._get_default_paras(), **kwargs)
        # logging.getLogger(__name__).debug("Dataset input paras:")
        # logging.getLogger(__name__).debug(self._paras)
        self._dataset_type = self._paras['dataset_type']

        # gather all needed parameters
        self._get_paras()
        self._prepare()
        logging.getLogger(__name__).debug(
            "DataSet end of __init__, para_string():")
        logging.getLogger(__name__).debug(self._para_string())

    def _get_default_paras(self):
        paras_def = {
            'is_shuffle': False,
            'is_pad': True,
            'is_cache': False,
            'is_lock': False,
            'epoch_max': -1
        }
        return paras_def

    def _get_paras(self):
        self._gather_paras_common()
        if self._dataset_type == "train":
            self._gather_paras_train()
        if self._dataset_type == "test":
            self._gather_paras_test()
        if self._dataset_type == "infer":
            self._gather_paras_infer()
        self._gather_paras_special()

    def _para_string(self):
        """Report all paras in a well formated string.
        May override to report used paras only.
        """
        # return json.dumps(self._paras, sort_keys=True, indent=4,
        #   separators=(',', ': '))
        # return self._paras
        dic_sorted = sorted(self._paras.items(), key=lambda t: t[0])
        fmt = r"{0}: {1},"
        msg = 'DataSet Settings:\n' + \
            '\n'.join([fmt.format(item[0], item[1]) for item in dic_sorted])
        return msg

    def _gather_paras_common(self):
        """Gather parameters from self._paras to shortcuts like self._batch_size.
        """

        # Input/Ouput tensor shapes
        self._batch_size = self._paras['batch_size']
        self._shape_i = self._paras['shape_i']
        self._shape_o = self._paras['shape_o']
        self._shape_sample_i = [1] + list(self._shape_i)
        self._shape_sample_o = [1] + list(self._shape_o)
        self._shape_batch_i = [self._batch_size] + list(self._shape_i)
        self._shape_batch_o = [self._batch_size] + list(self._shape_o)

        self._is_shuffle = self._paras['is_shuffle']
        self._epoch_max = self._paras['epoch_max']

        # Padding
        self._is_pad = self._paras['is_pad']

        # cache flag
        self._is_cache = self._paras['is_cache']

        # Dataset based on files
        self._is_lock = self._paras['is_lock']

    def _gather_paras_train(self):
        pass

    def _gather_paras_test(self):
        pass

    def _gather_paras_infer(self):
        pass

    def _gather_paras_special(self):
        pass

    def _prepare(self):
        """Prepare work before dataset is ready to use.
        """
        self._epoch = utp.Counter(max_state=self._epoch_max)

        # No more epoch flag is set to False only once here. When it is True,
        # Calling batch_start() will raise StopIteration exception.
        self._is_next_epoch = True
        self._is_next_epoch_to_be_true = False
        self._is_no_more_epoch = False

        # Work with self._is_lock, when self._is_next_file is False, Calling
        # _end_of_sinle_file() will results _padding(), otherwise it will load
        # next file and return sample()

        self._is_next_file = True
        self._is_next_file_to_be_true = False
        self._is_no_more_sample = False

    def _next_epoch(self):
        """prepare for next epoch
        If this method returns None, _sample() should work properly.
        If is_next_epoch is True, calling this will add one to epoch counter.
        Args:
            None
        Returns:
            None
        Raises:
            NoMoreEpoch: epoch_max reached
                - padding current batch
                - calling batch_start() will raise StopIteration
        """
        try:
            next(self._epoch.out)
        except StopIteration:
            raise NoMoreEpoch

    def __end_of_epoch(self):
        """called when self._sample() or self._next_file() raise a EndEpoch
        exception.
        Raises:
            ValueError : the new epoch is empty.
        """
        if self._is_next_epoch:
            try:
                self._next_epoch()
                return self._sample()
            except EndSingleFile:
                raise ValueError("Empty epoch.")
            except NoMoreEpoch:
                self._is_no_more_epoch = True
                return self._padding()
        else:
            return self._padding()

    def _next_file(self):
        """prepare for next file
        Raises:
            EndEpoch
        """
        pass

    def __end_of_single_file(self):
        """called when self._sample() raise a EndSingleFile exception
        """
        self._is_no_more_sample = True
        if self._is_next_file:
            try:
                self._next_file()
                self._is_no_more_sample = False
                return self._sample()
            except EndEpoch:
                self.__end_of_epoch()
        else:
            return self._padding()

    def _batch_start(self):
        """Called at start of next_batch()
        Raises:
            StopIteration
        """
        if self._is_no_more_sample:
            raise StopIteration
        if self._is_no_more_epoch:
            raise StopIteration

    def _batch_end(self):
        """Called at end of next_batch()
        """
        self._is_batch_end = True
        if self._is_next_file_to_be_true:
            self._is_next_file = True
            self._is_next_file_to_be_true = False
        if self._is_next_epoch_to_be_true:
            self._is_next_epoch = True
            self._is_next_epoch_to_be_true = False

    def _is_train_or_test(self):
        return self._dataset_type == "train" or self._dataset_type == "test"

    def _padding(self):
        """padding batch data when used up all datas and a batch is not finished
        Args:
        Returns:
            if self._is_pad:
                [1, shape_i], [1, shape_o] padding tensor
            or
                None, None
        Raises:
        """
        if not self._is_pad:
            return None, None
        else:
            return np.zeros(self._shape_sample_i), np.zeros(self._shape_sample_o)

    def _sample(self):
        """Returns two tensors of shape [1, shape_i] and [1, shape_o]
        """
        return np.zeros(self._shape_sample_i), np.zeros(self._shape_sample_o)

    def next_batch(self):
        """Returns next batch of current dataset.
        """

        self._batch_start()

        data_shape = [self.batch_size] + list(self._shape_i)
        label_shape = [self.batch_size] + list(self._shape_o)
        tensor_data = np.zeros(data_shape)
        tensor_label = np.zeros(label_shape)
        for i in range(self.batch_size):
            try:
                data, label = self._sample()
            except EndEpoch:
                data, label = self.__end_of_epoch()
            except EndSingleFile:
                data, label = self.__end_of_single_file()
            if data is None:
                tensor_data = tensor_data[:i]
                tensor_label = tensor_label[:i]
                break
            else:
                tensor_data[i, :] = data
                tensor_label[i, :] = label

        self._batch_end()

        if self._is_train_or_test:
            return tensor_data, tensor_label
        else:
            return tensor_data

    def print_paras(self):
        """Print all config parameters.
        """
        print(self._para_string())

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
        """maximum epoch"""
        return self._epoch.max_state

    @property
    def is_shuffle(self):
        """is shuffle"""
        return self._is_shuffle

    @property
    def is_pad(self):
        """is pad to fill batch"""
        return self._is_pad

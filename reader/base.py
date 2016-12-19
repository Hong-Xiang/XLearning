"""base class of datasets
"""
import logging
import numpy as np
import xlearn.utils.general as utg
import xlearn.utils.xpipes as utp


class DataSet(object):
    """Base class of all dataset classes.
    """

    # def __init__(self, filenames=None, **kwargs):
    def __init__(self, filenames=None, **kwargs):
        """Initialization of DataSet class.

        This base class handles reading parameters, epoch updates and batch
        generation. Inherence class should implement following method:

        Parameters:
        Required:
            None
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

        Parameter setting order:
            file[0] < file[1] < ... < file[-1] < kwargs
            not None paramters will be overwritten.

        Batch generation:
        Required:
            _sample(self)

        Optional:
            _padding(self)
            _triger_stop_iteration(self):
                By default, it will return self._padding()
                Returns: [1, *], [1, *] tensor

            _batch_start(self)
            _batch_end(self)
        """
        print("base init called.")
        # self._paras = {}
        # # set default parameters
        # self._paras = utg.merge_settings(
        #     self._paras, **self._set_paras_default())
        # # load parameters from configure files
        # if filenames is None:
        #     filenames = []
        # for filename in filenames:
        #     self._paras = self._load_conf_file(filename)

        # # set paramters directly from kwargs
        # self._paras = utg.merge_settings(self._paras, **kwargs)
        self._paras = utg.merge_settings(
            filenames=filenames, default_settings=self._get_default_paras(), **kwargs)
        self._dataset_type = self._paras['dataset_type']

        # gather all needed parameters
        self._get_paras()
        logging.getLogger(__name__).debug("DataSetBase __init__")
        logging.getLogger(__name__).debug(self._print_paras())

        self._prepare()

    def _get_default_paras(self):
        paras_def = {}
        paras_def.update({'is_shuffle': False})
        paras_def.update({'is_pad': True})
        paras_def.update({'is_cache': False})
        paras_def.update({'epoch_max': -1})
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

    def _print_paras(self):
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
        self._batch_size = self._paras['batch_size']
        self._shape_i = self._paras['shape_i']
        self._shape_o = self._paras['shape_o']
        self._is_pad = self._paras['is_pad']
        self._is_cache = self._paras['is_cache']
        self._epoch_max = self._paras['epoch_max']
        self._is_shuffle = self._paras['is_shuffle']

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

    def _next_epoch(self):
        """prepare for next epoch
        Args:
        Returns:
        Raises:
            StopIteration: epoch_max reached, or not yet for next epoch.
        """
        next(self._epoch.out)

    def _batch_start(self):
        """Called at start of next_batch()
        """
        pass

    def _batch_end(self):
        """Called at end of next_batch()
        """
        pass

    def _is_train_or_test(self):
        return self._dataset_type == "train" or self._dataset_type == "test"

    def _triger_stop_iteration(self):
        """called when self._sample() raise a StopIteration Exception.
        """
        try:
            self._next_epoch()
        except StopIteration:
            return self._padding()
        return self._sample()

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
            return self._sample()

    def _sample(self):
        """Returns two tensors of shape [1, shape_i] and [1, shape_o]
        """
        return np.zeros(self._shape_i), np.zeros(self._shape_o)

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
            except StopIteration:
                data, label = self._triger_stop_iteration()
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
        print(self._print_paras)

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

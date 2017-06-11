""" base class for samples generator for Keras models."""
# TODO: batch mode?
# TODO: Add tensorflow queue support?
# TODO: Add RBG support
import os
from collections import defaultdict, namedtuple
from enum import Enum
import json
import random
import numpy as np
import h5py
import pathlib
import tensorflow as tf
from ..utils.prints import pp_json
from xlearn.utils.collections import AttrDict
from xlearn.utils.options import json_config, auto_configs

import xlearn.datasets.rawfile as rf
import xlearn.datasets.loaders as ldr
import xlearn.datasets.processors as pr

from ..utils.tensor import downsample
from ..utils.cells import Sampler


PATH_DATASETS = os.environ['PATH_DATASETS']
PATH_XLEARN = os.environ['PATH_XLEARN']


# class DatasetConstructor:
#     @json_config
#     @auto_configs
#     def __init__(self,
#                  dataset_name,
#                  raw_file_type_names,                 
#                  raw_file_kwarges)
#         pass
    
#     def print_configs(self):
#         pp_json(dict(self.c), "DatasetConstructor Configs")

#     def new_dataset():
#         raw_file_readers = dict()
#         for rn, rkwargs in zip(self.c.raw_file_type_names, self.c.raw_file_kwarges):
#             if rkwargs is None:
#                 rkwargs = dict()    
#             rr = rf.get_instance(rn, **rkwargs)
#             raw_file_readers[rr.name] = rr

class Mode(Enum):
    TRAIN = "train"
    TEST = "test"

class RawdataAuto:        
    def __init__(self, raw_file_name, keys, mode, nb_thread=4):
        path_db = pathlib.Path(PATH_DATASETS)
        if str(raw_file_name).endswith('.h5'):
            self._raw_file = rf.HDF5(path_db/raw_file_name, keys)
        elif str(raw_file_name).endswith('.npz'):
            self._raw_file = rf.NPYZ(path_db/raw_file_name, keys)
        else:
            raise ValueError("Raw file name {fn} not understand.".format(fn=raw_file_name))

        # !!! HIGH RISK !!!
        idx_file = raw_file_name[:raw_file_name.rfind('.')]+'.idx.'+mode+'.npy'
        #

        self._idx_sampler = ldr.IndexSampler(is_shuffle=True, idx_file=str(path_db/idx_file))        
        self.nb_thread = nb_thread
        

        self._raw_loader = [ldr.Loader(datasets=self._raw_file.datasets, idx_sampler=self._idx_sampler) for i in range(self.nb_thread)]
        self.__c_thread = 0

    def init(self):
        self._raw_file.init()

    def final(self):
        self._raw_file.final()
    
    @property
    def datasets(self):
        return self._raw_file.datasets

    def _generate(self):
        self.__c_thread += 1
        self.__c_thread = self.__c_thread % self.nb_thread
        return next(self._raw_loader[self.__c_thread])
    
    def sample(self):
        return self._generate()
    
    def __next__(self):
        return self.sample()

        


class Dataset:
    """ base class of general dataset

    # Sample datas:

    1.  `sample()`
    2.  `__next__()`
    3.  `sess.run(sample_op)`

    Initialization

        self._initialize()
        self.__enter__()

    Finalize

        self.__exit__()

        file_name: filename of dataset file, could be None if uses generative dataset.
        file_type: one of "h5", "npy", "gen".
    """
    dkeys = ('data', 'idx')
    dtypes = (np.float32, np.int64)
    @json_config
    @auto_configs()
    def __init__(self,
                 batch_size,
                 mode,                            
                 raw_file_name=None,
                 raw_file_keys=None,                 
                 proc_names=tuple(),
                 proc_kwargs=tuple(),
                 is_tf=False,
                 queue_threads=4,
                 name='dataset'):
        """ Dataset class
        Support basic loading and processing.
        
        
        """
        self.mode = mode

        # A dict of sliceable object.
        self.raw_datas = RawdataAuto(raw_file_name, raw_file_keys, mode)

        if not hasattr(self, 'c'):
            self.c = AttrDict()
        self.shapes = dict()
        self.c['dkeys'] = self.dkeys
        self.c['dtypes'] = self.dtypes
        self.c['dshapes'] = self.shapes

    def init(self):
        self.raw_datas.init()
        self._init_impl()
        if self.c.is_tf:
            self._create_ops()
        self._datasets_checks()
        self.__is_init = True
        self.print_settings()

    def __enter__(self):
        self.init()
        return self

    def final(self):
        self._final_impl()
        self.raw_datas.final()
        self.__is_init = False

    def __exit__(self, etype, value, traceback):
        self.final()

    def _bind_procs(self):
        # Automatically determin absolute path of raw data file.        
        self.procs = list()
        for pn, pkwargs in zip(self.c.proc_names, self.c.proc_kwargs):
            self.procs.append(pr.get_instance(pn, **pkwargs))

    def _infer_shapes(self):
        raw_shapes = dict()
        for k in self.raw_datas.datasets:            
            raw_shapes[k] = self.raw_datas.datasets[k].shape[1:]
        raw_shapes['idx'] = []
        if isinstance(self.procs[-1], pr.Proc):
            data_shapes = self.procs[-1](raw_shapes)
        else:
            data_shapes = raw_shapes
        for k in self.dkeys:
            self.shapes[k] = data_shapes[k]

    def _init_impl(self):
        self._bind_procs()
        self._infer_shapes()

    def _final_impl(self):
        pass

    def _datasets_checks(self):
        pass

    def _generate(self):
        s = self.raw_datas.sample()
        for p in reversed(self.procs):
            s = p(s)
        return [s[k] for k in self.dkeys]

    def sample(self):
        """ Genreate a new sample
        Returns:
            A dict of mini-batch tensors.
        """
        if not self.__is_init:
            raise ValueError('Not initilized.')
        out = defaultdict(list)

        # sample single
        for i in range(self.c.batch_size):
            g = self._generate()
            for i, k in enumerate(self.dkeys):
                out[k].append(g[i])
        # convert to numpy
        for k in self.dkeys:
            out[k] = np.array(out[k])
        return out

    def __next__(self):
        return self.sample()

    def __iter__(self):
        return self

    def print_settings(self):
        fmt = r"DATASET {name}:{mode} SETTINGS:"
        title = fmt.format(name=self.c.name, mode=self.mode)        
        pp_json(dict(self.c), title=title)

    def _create_ops(self):
        import tensorflow as tf
        with tf.name_scope("dataset/{name}/{mode}".format(name=self.c.name, mode=self.c.mode)):
            self.__single_op = tf.py_func(
                self._generate, [], self.dtypes, name="single_op", stateful=True)
            queue = tf.FIFOQueue(capacity=self.c.queue_threads*self.c.batch_size,
                                 dtypes=self.dtypes,
                                 shapes=[self.shapes[k] for k in self.dkeys],
                                 name="queue")
            enqueue_op = queue.enqueue(self.__single_op)
            self.__sample_op = queue.dequeue_many(self.c.batch_size)
            queue_runner = tf.train.QueueRunner(
                queue, [enqueue_op] * self.c.queue_threads)
            tf.add_to_collection(tf.GraphKeys.QUEUE_RUNNERS, queue_runner)
            self.nodes = AttrDict()
            for i, k in enumerate(self.dkeys):
                self.nodes[k] = self.__sample_op[i]

    @property
    def sample_single_op(self):
        return self.__single_op

    @property
    def sample_op(self):
        return self.__sample_op

    def sample_many(self, nb_examples):
        """ gather given numbers of examples """
        nb_samples = int(numpy.ceil(nb_examples / self.p.batch_size))
        out = defaultdict(list)
        for _ in range(nb_samples):
            s = self.sample()
            for k in s:
                out[k].append(s[k])
        for k in out:
            out[k] = np.concatenate(out[k], axis=0)
            out[k] = out[k][:nb_examples, ...]
        return out

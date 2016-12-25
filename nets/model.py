"""
Definition of subnets.

Subnets must have following methods:
output = infer(input[s])
output = loss(input[s], lable[s])
"""
# TODO: Add support for:
# variable_list = variables([flags])

import os
import json
import logging
import numpy as np
import tensorflow as tf
import xlearn.utils.general as utg
import xlearn.nets.layers as layer
import xlearn.utils.tensor as ut


FLAGS = tf.app.flags.FLAGS
DEFAULT_CONF_JSON = os.getenv(
    'DEFAULT_NET_CONF', "/home/hongxwing/Workspace/xlearn/nets/conf_net.json")


def def_flag(var_type, name, value, helpdoc=None):
    """Warper functions to define multi-type flags.
    :param var_type: type of flag, in string.
    :param name: name of flag

    """
    flag = tf.app.flags
    if var_type == 'int':
        flag.DEFINE_integer(name, value, helpdoc)
    if var_type == 'float':
        flag.DEFINE_float(name, value, helpdoc)
    if var_type == 'string':
        flag.DEFINE_string(name, value, helpdoc)
    if var_type == 'bool':
        flag.DEFINE_boolean(name, value, helpdoc)


def define_flags(filenames=None, **kwargs):
    """define FLAGS from JSON files
    """
    if filenames is None:
        filenames = [DEFAULT_CONF_JSON]
    flag = tf.app.flags
    logging.getLogger(__name__).info(
        "Load tf.app.flags config file: {}".format(filenames))
    args = utg.merge_settings(filenames=filenames, **kwargs)
    for item in args:
        if not isinstance(args[item], dict):
            continue
        if 'helpdoc' not in args[item]:
            continue
        flag_type = args[item]['type']
        flag_name = item
        flag_value = args[item]['value']
        flag_help = args[item]['helpdoc']
        def_flag(flag_type, flag_name, flag_value, flag_help)
    try:
        FLAGS.activation_function
    except AttributeError:
        def_flag("string", "activation_function", "elu", "default activation.")


def before_net_definition():
    zeroinit = tf.constant_initializer(0.0)
    with tf.variable_scope('net_global') as scope:
        global_step = tf.get_variable('global_step',
                                      shape=[],
                                      trainable=False,
                                      initializer=zeroinit,
                                      dtype=tf.float32)


def scalar_summary(x):
    """Helper to create scalar summaries.
    :param x:
    :return: nothing
    """
    tensor_name = x.op.name
    tf.summary.scalar(tensor_name, x)


def activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


class NetManager(object):
    """Handling interaction with nets.
    """

    def __init__(self, net, varnames=None):
        self._net = net
        self._summary = tf.summary.merge_all()
        # self._sess = tf.Session(
        # config=tf.ConfigProto(log_device_placement=True))
        self._sess = tf.Session()
        init_op = tf.global_variables_initializer()
        self._sess.run(init_op)
        self._train_writer = tf.summary.FileWriter(
            FLAGS.path_summary_train, self._sess.graph)
        self._test_writer = tf.summary.FileWriter(
            FLAGS.path_summary_test, self._sess.graph)

        if varnames is None:
            self._saver = tf.train.Saver(tf.global_variables())
        else:
            self._saver = tf.train.Saver(varnames)
        if FLAGS.is_restore:
            logging.getLogger(__name__).info("Restoring Variables...")
            self.restore()

    def run(self, tensor_list, feed_dict):
        return self._sess.run(tensor_list, feed_dict)

    def write_summary(self, feed_dict, is_test=False):
        summaries = self._sess.run(self._summary, feed_dict=feed_dict)
        step = self._net.global_step(self._sess)
        if is_test:
            self._test_writer.add_summary(summaries, step)
        else:
            self._train_writer.add_summary(summaries, step)
        return summaries

    def save(self, step=None):
        if step is None:
            self._saver.save(self._sess, FLAGS.path_save + '-' +
                             FLAGS.run_task + '-', global_step=net.global_step(self._sess))
        else:
            self._saver.save(self._sess, FLAGS.path_save +
                             '-' + FLAGS.run_task, step)

    def restore(self):
        if self._net.is_skip_restore:
            return
        ckfile = os.path.join(FLAGS.path_save, 'checkpoint')
        with open(ckfile) as ckpf:
            for line in ckpf:
                (key, value) = line.split()
                if key == 'model_checkpoint_path:':
                    save_path = value
                    break
        save_path = save_path[1:-1]
        save_path = os.path.join(FLAGS.path_save, save_path)
        self._saver.restore(self._sess, save_path)

    @property
    def sess(self):
        return self._sess


class TFNet(object):

    def __init__(self, filenames=None, name='TFNet', varscope=None, **kwargs):
        logging.getLogger(__name__).debug('filenames:{}'.format(filenames))
        self._name = name
        self._paras = utg.merge_settings(filenames=filenames, **kwargs)
        if varscope is None:
            varscope = tf.get_variable_scope()
        self._var_scope = varscope
        self._input = None
        self._label = None
        self._infer = None
        self._loss = None
        self._train = None

        with tf.variable_scope('net_global') as scope:
            scope.reuse_variables()
            self._global_step = tf.get_variable('global_step', trainable=False)
        self._learn_rate = tf.train.exponential_decay(FLAGS.lr_init,
                                                      self._global_step,
                                                      FLAGS.lr_decay_steps,
                                                      FLAGS.lr_decay_factor,
                                                      staircase=True,
                                                      name='learning_rate')
        tf.summary.scalar("learn_rate", self._learn_rate)
        self._gather_paras()
        self._prepare()
        logging.getLogger(__name__).debug(
            "TFNet end of __init__, para_string():")
        logging.getLogger(__name__).debug(self._para_string())

    def _gather_paras(self):
        pass

    def _prepare(self):
        self._is_skip_restore = False
        pass

    def _net_definition(self):
        pass

    def global_step(self, sess):
        return sess.run(self._global_step)

    def _para_string(self):
        dic_sorted = sorted(self._paras.items(), key=lambda t: t[0])
        fmt = r"{0}: {1},"
        msg = 'DataSet Settings:\n' + \
            '\n'.join([fmt.format(item[0], item[1]) for item in dic_sorted])
        return msg

    @property
    def infer(self):
        return self._infer

    @property
    def loss(self):
        return self._loss

    @property
    def inputs(self):
        return self._input

    @property
    def label(self):
        return self._label

    @property
    def train(self):
        return self._train

    @property
    def learn_rate(self):
        return self._learn_rate

    @property
    def is_skip_restore(self):
        return self._is_skip_restore

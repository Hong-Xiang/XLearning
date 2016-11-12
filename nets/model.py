"""
Definition of subnets.

Subnets must have following methods:
output = infer(input[s])
output = loss(input[s], lable[s])
variable_list = variables([flags])
"""
from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
import xlearn.nets.layers as layer

FLAGS = tf.app.flags.FLAGS


def define_flags():
    flag = tf.app.flags
    flag.DEFINE_float("weight_decay", 0.0001,
                      "Weight decay coefficient.")
    flag.DEFINE_float("eps", 1e-5,
                      "Weight decay coefficient.")
    flag.DEFINE_integer("batch_size", 64, "Batch size.")
    flag.DEFINE_integer("hidden_units", 64, "# of hidden units.")
    flag.DEFINE_float("lr_init", None, "Initial learning rate.")
    flag.DEFINE_string("save_dir", '.', "saving path.")
    flag.DEFINE_string("save_path", 'netsave', "saving path.")
    flag.DEFINE_string("summary_dir", '.', "summary path.")
    flag.DEFINE_integer("lr_decay_steps", 1000, "decay steps.")
    flag.DEFINE_float("lr_decay_factor", 0.6,
                      "learing rate decay factor.")
    flag.DEFINE_integer("height", 11, "patch_height")
    flag.DEFINE_integer("width", 11, "patch_width")
    flag.DEFINE_integer("down_ratio", 3, "down_sample_ratio")
    flag.DEFINE_integer("patch_per_file", 32, "patches per file.")
    flag.DEFINE_string("train_path", None, "train data path.")
    flag.DEFINE_string("test_path", None, "test data path.")
    flag.DEFINE_string("prefix", None, "prefix of data files.")
    flag.DEFINE_float("leak_ratio", None, "lrelu constant.")
    flag.DEFINE_integer("hidden_layer", 10, "hidden layers")
    flag.DEFINE_string("task", None, "test task.")
    flag.DEFINE_string("grad_clip", 100, "maximum gradient value.")
    flag.DEFINE_boolean("restore", False, "restore variables.")
    flag.DEFINE_integer("steps", 1000, "train steps.")
    flag.DEFINE_boolean("is_train", True, "flag of is training.")
    flag.DEFINE_boolean("only_down_width", False, "flag of only downsample width")

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
    tf.scalar_summary(tensor_name, x)


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
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


class NetManager(object):
    """Handling interaction with nets.
    """

    def __init__(self, net, varnames=None):
        self._net = net
        self._summary = tf.merge_all_summaries()
        self._sess = tf.Session()
        init_op = tf.initialize_all_variables()
        self._sess.run(init_op)
        self._summary_writer = tf.train.SummaryWriter(
            FLAGS.summary_dir, self._sess.graph)

        if varnames is None:
            self._saver = tf.train.Saver(tf.all_variables())
        else:
            self._saver = tf.train.Saver(varnames)
        if FLAGS.restore:
            print("restoring variables.")
            self.restore()

    def run(self, tensor_list, feed_dict):
        return self._sess.run(tensor_list, feed_dict)

    def write_summary(self, feed_dict):
        summaries = self._sess.run(self._summary, feed_dict=feed_dict)
        step = self._net.global_step(self._sess)
        self._summary_writer.add_summary(summaries, step)

    def save(self, step=None):
        if step is None:
            self._saver.save(self._sess, FLAGS.save_dir+'-'+FLAGS.task)
        else:
            self._saver.save(self._sess, FLAGS.save_dir+'-'+FLAGS.task, step)

    def restore(self):
        ckfile = os.path.join(FLAGS.save_dir, 'checkpoint')
        with open(ckfile) as ckpf:
            for line in ckpf:
                (key, value) = line.split()
                if key == 'all_model_checkpoint_paths:':
                    save_path = value
                    break
        save_path = save_path[1:-1]
        save_path = os.path.join(FLAGS.save_dir, save_path)
        self._saver.restore(self._sess, save_path)

    @property
    def sess(self):
        return self._sess


class TFNet(object):

    def __init__(self, varscope=tf.get_variable_scope()):
        self._var_scope = varscope
        self._summary_writer = None
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
        tf.scalar_summary("learn_rate", self._learn_rate)

    def _net_definition(self):
        pass

    def global_step(self, sess):
        return sess.run(self._global_step)

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
    def summary_writer(self):
        return self._summary_writer

    @property
    def learn_rate(self):
        return self._learn_rate

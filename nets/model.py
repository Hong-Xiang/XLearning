"""
Definition of subnets.

Subnets must have following methods:
output = infer(input[s])
output = loss(input[s], lable[s])
variable_list = variables([flags])
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import xlearn.nets.layers as layer

FLAGS = tf.app.flags.FLAGS
def before_net_definition():
    zeroinit = tf.constant_initializer(0.0)
    with tf.variable_scope('net_global') as scope:
        global_step = tf.get_variable('global_step',
                                      shape=[],
                                      trainable=False,
                                      initializer=zeroinit,
                                      dtype=tf.float32)

def scalar_summary(x):
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
    def __init__(self, net, varnames=None):
        self._net = net
        self._summary = tf.merge_all_summaries()
        self._sess = tf.Session()
        init_op = tf.initialize_all_variables()
        if varnames is None:
            self._saver = tf.train.Saver(tf.all_variables())
        else:
            pass
        self._summary_writer = tf.train.SummaryWriter(FLAGS.summary_dir, self._sess.graph)
        self._sess.run(init_op)
    
    def run(self, tensor_list, feed_dict):
        return self._sess.run(tensor_list, feed_dict)

    def write_summary(self, feed_dict):
        summaries = self._sess.run(self._summary, feed_dict=feed_dict)
        step = self._net.global_step(self._sess)
        self._summary_writer.add_summary(summaries, step)
    
    def save(self, step=None):
        if step is None:
            step = self._net.global_setp(self._sess)
        saver.save(self._sess, FLAGS.save_dir, step)
    
    def restore(self):
        saver.restore(self._sess, FLAGS.save_dir)

    @property
    def sess(self):
        return self._sess        

class TFNet(object):
    def __init__(self, varscope=tf.get_variable_scope()):
        self._var_scope = varscope
        self._summary_writer = None
        self._input = None
        self._label = None
        self._loss = None
        self._train = None        

        with tf.variable_scope('net_global') as scope:
            scope.reuse_variables()
            self._global_step = tf.get_variable('global_step', trainable=False)
        self._learn_rate = tf.train.exponential_decay(FLAGS.learning_rate_init,
                                            self._global_step,
                                            FLAGS.decay_steps,
                                            FLAGS.learning_rate_decay_factor,
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

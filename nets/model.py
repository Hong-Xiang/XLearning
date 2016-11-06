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
            self._global_step = tf.get_variable('global_step')

    def _net_definition(self):
        pass        
    
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

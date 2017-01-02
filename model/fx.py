"""net for infer f(x) with given x in R^1
"""
from __future__ import absolute_import, division, print_function
from six.moves import xrange
import tensorflow as tf
import numpy as np
import xlearn.nets.layers as layer
import xlearn.nets.model as model
from xlearn.nets.model import TFNet

FLAGS = tf.app.flags.FLAGS


class NetFx(TFNet):
    """Net to infer a function to x
    """

    def __init__(self, filenames=None, name='FxNet', varscope=None, **kwargs):
        super(NetFx, self).__init__(filenames=filenames,
                                    name=name, varscope=varscope, **kwargs)
        self._input = layer.inputs([self._batch_size, 1])
        self._label = layer.labels([self._batch_size, 1])
        self._net_definition()
        self._add_summary()
    
    def _gather_paras(self):
        self._batch_size = self._paras['batch_size']
        self._is_possion_layer = self._paras['is_possion_layer']
        self._is_dropout = self._paras['is_dropout']
        self._is_bayes = self._paras['is_bayes']

    def _net_definition(self):
        self._keep = tf.placeholder(name="keep_prop", shape=[], dtype=tf.float32)
        hidden = layer.full_connect(
            self._input, [1, FLAGS.hidden_units], name='hidden0')
        self._midops = []
        for i in range(1, FLAGS.hidden_layer + 1):
            if self._is_dropout:
                hidden = tf.nn.dropout(hidden, keep_prob=self._keep)
            hidden = layer.matmul_bias(
                hidden, [FLAGS.hidden_units, FLAGS.hidden_units], name='hidden%d' % i)
            # hidden = layer.batch_norm(hidden)
            hidden = layer.activation(hidden)
            self._midops.append(hidden)

        self._infer = layer.matmul_bias(
            hidden, [FLAGS.hidden_units, 1], name='infer')
        if self._is_possion_layer:
            end_layer = layer.possion_layer(self._infer)
        else:
            end_layer = tf.identity(self._infer)
        ones = np.ones([self._batch_size, 1])
        sigma = tf.constant(ones, dtype=tf.float32)


        if self._is_bayes:
            self._loss = layer.gaussian_loss(end_layer, sigma, self._input)
        else:
            self._loss = layer.l2_loss(end_layer, self._label)
        self._train = layer.trainstep(
            self._loss, self._learn_rate, self._global_step)

    def _add_summary(self):
        model.scalar_summary(self._loss)
        for op in self._midops:
            model.activation_summary(op)
    
    @property
    def keep_prob(self):
        return self._keep

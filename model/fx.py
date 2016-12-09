"""net for infer f(x) with given x in R^1
"""
from __future__ import absolute_import, division, print_function
from six.moves import xrange
import tensorflow as tf
import xlearn.nets.layers as layer
import xlearn.nets.model as model
from xlearn.nets.model import TFNet

FLAGS = tf.app.flags.FLAGS


class NetFx(TFNet):
    """Net to infer a function to x
    """

    def __init__(self, batch_size, varscope=tf.get_variable_scope()):
        super(NetFx, self).__init__(varscope)
        self._batch_size = batch_size
        self._input = layer.inputs([self._batch_size, 1])
        self._label = layer.labels([self._batch_size, 1])
        self._net_definition()
        self._add_summary()

    def _net_definition(self):
        hidden = layer.full_connect(
            self._input, [1, FLAGS.hidden_units], name='hidden0')
        self._midops = []
        for _ in xrange(1, FLAGS.hidden_layer + 1):
            hidden = layer.matmul_bias(
                hidden, [FLAGS.hidden_units, FLAGS.hidden_units])
            # hidden = layer.batch_norm(hidden)
            hidden = layer.activation(hidden)
            self._midops.append(hidden)

        self._infer = layer.matmul_bias(
            hidden, [FLAGS.hidden_units, 1], name='infer')
        self._loss = layer.l2_loss(self._infer, self._label)
        self._train = layer.trainstep(
            self._loss, self._learn_rate, self._global_step)

    def _add_summary(self):
        model.scalar_summary(self._loss)
        for op in self._midops:
            model.activation_summary(op)

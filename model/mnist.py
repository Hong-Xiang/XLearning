"""
MNIST nets.

Subnets must have following methods:
output = infer(input[s])
output = loss(input[s], lable[s])
variable_list = variables([flags])
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import xlearn.nets.layers as layer
from xlearn.nets.model import TFNet 

class MNIST(TFNet):
    def __init__(self, n_hidden, varscope=tf.get_variable_scope()):
        super(MNIST, self).__init__(varscope=varscope)     
        height = 28
        width = 28
        self._pixels = height*width
        self._input = layer.inputs([None, height, width, 1])
        self._label = layer.labels([None, 10])
        self._n_hidden = n_hidden
        self._net_definition(self._var_scope)        
        
    def _net_definition(self, varscope):
        flaten = tf.reshape(self._input,
                            [-1, 28*28],
                            name='flatten')

        hidden = layer.full_connect(flaten,
                                    [self._pixels, self._n_hidden],
                                    name='hidden',
                                    varscope=varscope)
        
        
        hidden2 = layer.full_connect(hidden,
                                    [self._n_hidden, self._n_hidden],
                                    name='hidden',
                                    varscope=varscope)

        self._infer = layer.matmul_bias(hidden2,
                                  [self._n_hidden, 10],
                                  name='infer',
                                  varscope=varscope)
                        
        self._loss = layer.predict_loss(self._infer, self._label)        
        self._accuracy = layer.predict_accuracy(self._infer, self._label)
        self._train = tf.train.GradientDescentOptimizer(0.5).minimize(self._loss)
        
    @property
    def accuracy(self):
        return self._accuracy

class MNISTConv(TFNet):
    def __init__(self, n_hidden, varscope=tf.get_variable_scope()):
        super(MNISTConv, self).__init__(varscope=varscope)     
        height = 28
        width = 28
        self._pixels = height*width
        self._input = layer.inputs([None, height, width, 1])
        self._label = layer.labels([None, 10])
        self._n_hidden = n_hidden
        self._keep_prob = tf.placeholder(tf.float32)
        self._net_definition(self._var_scope)
        

    def _net_definition(self, varscope):        
        conv0 = layer.feature(self._input,
                              filter_shape=[5, 5, 1, 32],
                              strides_conv=[1,1,1,1],
                              padding_conv='SAME',
                              pooling_ksize=[1, 2, 2, 1],
                              strides_pool=[1, 2, 2, 1],
                              padding_pool = 'SAME',
                              name='conv0')

        conv1 = layer.feature(conv0,
                              filter_shape=[5, 5, 32, 64],
                              strides_conv=[1,1,1,1],
                              padding_conv='SAME',
                              pooling_ksize=[1, 2, 2, 1],
                              strides_pool=[1, 2, 2, 1],
                              padding_pool = 'SAME',
                              name='conv1')
        
        with tf.name_scope('flat') as scope:
            flat0 = tf.reshape(conv1, [-1, 7*7*64], name='reshape')
        
        full0 = layer.full_connect(flat0, shape=[7*7*64, self._n_hidden])

        drop0 = layer.dropout(full0, self._keep_prob)

        self._infer = layer.matmul_bias(drop0,
                                        [self._n_hidden, 10],
                                        name='infer',
                                        varscope=varscope)
                        
        self._loss = layer.predict_loss(self._infer, self._label)        
        self._accuracy = layer.predict_accuracy(self._infer, self._label)
        self._train = tf.train.AdamOptimizer(1e-4).minimize(self._loss)
        
    @property
    def accuracy(self):
        return self._accuracy
    
    @property
    def keep_prob(self):
        return self._keep_prob
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from ..keras_ext.constrains import MaxMinValue
from keras.layers.advanced_activations import ELU

NORMAL_ACTIVATIONS = ['softmax', 'softplus', 'relu',
                      'tanh', 'sigmoid', 'hard_sigmoid', 'linear']


def dense_stack(input_dim, output_dim, hiddens, activation='elu', last_activation=None, name='denses', is_bn=True, is_dropout=False, dropout_rate=0.5):
    """ sequential dense layers """
    n_hidden_layers = len(hiddens)
    with tf.name_scope(name):
        m = Sequential(name=name)
        for i in range(n_hidden_layers):
            with tf.name_scope('hidden_block'):
                with tf.name_scope('dense'):
                    if i == 0:
                        m.add(Dense(hiddens[i], activation=activation,
                                    input_shape=(input_dim,)))
                    else:
                        m.add(Dense(hiddens[i], activation=activation))
                if is_bn:
                    with tf.name_scope('bn'):
                        m.add(BatchNormalization())
                if is_dropout:
                    with tf.name_scope('dropout'):
                        m.add(Dropout(dropout_rate))
        with tf.name_scope('output'):
            if last_activation is None:
                m.add(Dense(output_dim, name=name + '/Dense_end'))
            else:
                m.add(Dense(output_dim, activation=last_activation,
                            name=name + '/Dense_end'))
    return m


def critic1d_wgan(input_dim, output_dim, hiddens, activation='elu', name='critic', is_bn=True, is_dropout=False, dropout_rate=0.5):
    """ critic1d for GANs """
    n_hidden_layers = len(hiddens)
    with tf.name_scope(name):
        m = Sequential(name=name)
        for i in range(n_hidden_layers):
            with tf.name_scope('hidden_block'):
                with tf.name_scope('dense'):
                    if i == 0:
                        m.add(Dense(hiddens[i], activation=activation,
                                    input_shape=(input_dim,), kernel_constraint=MaxMinValue()))
                    else:
                        m.add(
                            Dense(hiddens[i], activation=activation, kernel_constraint=MaxMinValue()))
                if is_bn:
                    with tf.name_scope('bn'):
                        m.add(BatchNormalization())
                if is_dropout:
                    with tf.name_scope('dropout'):
                        m.add(Dropout(dropout_rate))
        with tf.name_scope('output'):
            m.add(Dense(output_dim, kernel_constraint=MaxMinValue(),
                        name=name + '/Dense_end'))
    return m

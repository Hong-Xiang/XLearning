import tensorflow as tf
import numpy as np
from keras.layers import Convolution2D, BatchNormalization, Activation, Dense, Dropout
from keras.models import Model, Sequential

FLAGS = tf.app.flags.FLAGS


def Label(shape, name='label', dtype=tf.float32):
    shape = [None] + list(shape)
    return tf.placeholder(dtype, shape, name)


def Input(shape, name='input', dtype=tf.float32):
    shape = [None] + list(shape)
    return tf.placeholder(dtype, shape, name)


def Output(shape, name='output', dtype=tf.float32):
    shape = [None] + list(shape)
    return tf.placeholder(dtype, shape, name)


def Convolution2DwithBN(tensor_in, nb_filter, nb_row, nb_col, activation='elu', border_mode='same', is_summary=True, name=None):
    with tf.name_scope(name):
        x = tensor_in
        x = Convolution2D(nb_filter, nb_row, nb_col,
                          border_mode=border_mode, name='convolution')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
    return x


def Denses(input_shape, output_dim, hiddens, activation='elu', last_activation=None, name='denses', is_bn=True, is_dropout=False, dropout_rate=0.5):
    """ sequential dense layers """
    n_hidden_layers = len(hiddens)
    with tf.name_scope(name):
        m = Sequential(name=name)
        for i in range(n_hidden_layers):
            with tf.name_scope('hidden_block'):
                with tf.name_scope('dense'):
                    if i == 0:
                        m.add(Dense(hiddens[i], activation=activation,
                                    input_shape=input_shape))
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

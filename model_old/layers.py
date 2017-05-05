import tensorflow as tf
import tensorflow.contrib.layers as ly
import numpy as np
from keras.layers import Convolution2D, BatchNormalization, Activation, Dense, Dropout
from keras.models import Model, Sequential
from . import config

FLAGS = tf.app.flags.FLAGS


def Label(shape, name='label', dtype=tf.float16):
    shape = [None] + list(shape)
    return tf.placeholder(dtype, shape, name)


def Input(shape, name='input', dtype=tf.float16):
    shape = [None] + list(shape)
    return tf.placeholder(dtype, shape, name)


def Latent(shape, name='latent', dtype=tf.float16):
    shape = [None] + list(shape)
    return tf.placeholder(dtype, shape, name)


def Condition(shape, name='condition', dtype=tf.float16):
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


def Denses(input_shape, output_dim, hiddens, activation='elu', last_activation=None, name='denses', is_reuse=False, is_bn=True, is_dropout=False, dropout_rate=0.5):
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


def dense_stack(inputs,
                num_outputs,
                hiddens,
                hidden_activation_fn=tf.nn.elu,
                last_activation_fn=None,
                is_bn=False,
                is_dropout=False,
                keep_prob=0.5,
                is_reuse=False,
                name_scope='dense_stack',
                var_scope='dense_stack_vars'):
    """ Stack of fc layers
    Args:
    Returns:
    """
    n_hidden_layers = len(hiddens)
    with tf.name_scope(name_scope):
        with tf.variable_scope(var_scope) as scope:
            if is_reuse:
                scope.reuse_variables()
            var_scope_hidden = []
            for i in range(n_hidden_layers):
                with tf.variable_scope('fc_%d' % i) as scope_h:
                    var_scope_hidden.append(scope_h)
            with tf.variable_scope('fc_end') as scope_l:
                var_scope_last = scope_l
            x = inputs
            normalizer_fn = ly.batch_norm if is_bn else None
            biases_initializer = tf.constant_initializer()
            cly = 0
            for i in range(n_hidden_layers):
                x = ly.fully_connected(inputs=x,
                                       num_outputs=hiddens[i],
                                       reuse=is_reuse,
                                       scope=var_scope_hidden[i],
                                       normalizer_fn=normalizer_fn,
                                       biases_initializer=biases_initializer)
                if is_dropout:
                    x = ly.dropout(inputs=x,
                                   keep_prob=keep_prob,
                                   is_training=config.is_train,
                                   scope=var_scope_hidden[i])
                if hidden_activation_fn is not None:
                    x = hidden_activation_fn(x)
                cly += 1
            x = ly.fully_connected(inputs=x,
                                   num_outputs=num_outputs,
                                   activation_fn=last_activation_fn,
                                   reuse=is_reuse,
                                   scope=var_scope_last,
                                   biases_initializer=biases_initializer)
            if last_activation_fn is not None:
                x = last_activation_fn(x)
    return x

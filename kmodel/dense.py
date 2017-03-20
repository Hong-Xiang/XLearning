import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.layers.advanced_activations import ELU

NORMAL_ACTIVATIONS = ['softmax', 'softplus', 'relu',
                      'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

def dense_stack(input_dim, output_dim, hiddens, activation='elu', last_activation=None, name='denses', is_reuse=False, is_bn=True, is_dropout=False, dropout_rate=0.5):
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

# def denses(ip=None, input_shape=None, hiddens=None, is_dropout=True, activation='elu', name='denses', dropout_rate=0.5, **kwargs):
#     """ sequential dense layers """
#     if hiddens is None:
#         hiddens = [128]
#     n_hidden_layers = len(hiddens) 
#     if ip is not None:
#         input_shape = ip.get_shape()[1:].as_list()        
#     m = Sequential(name=name)
#     for i in range(n_hidden_layers):
#         if i == 0:
#             m.add(Dense(hiddens[i], activation=activation,
#                         input_shape=input_shape, name=name + '/Dense_%d' % i))
#         else:
#             m.add(Dense(hiddens[i], activation=activation,
#                         name=name + '/Dense_%d' % i))
#         if is_dropout:
#             m.add(Dropout(dropout_rate, name=name + '/Dropout_%d' % i))
#     if ip is not None:
#         output = m(ip)
#     else:
#         output = m
#     return output


def critic(ip=None, input_shape=None, hiddens=None, is_dropout=True, activation='elu', name='critic_dense', dropout_rate=0.5, **kwrags):
    m = Sequential(name=name)
    if ip is not None:
        input_shape = ip.get_shape()[1:].as_list()
    m.add(denses(input_shape=input_shape, hiddens=hiddens, is_dropout=is_dropout,
                 activation=activation, dropout_rate=dropout_rate, name='critic_denses', **kwrags))
    m.add(Dense(1, activation='sigmoid', name=name + '/Sigmoid'))
    if ip is not None:
        output = m(ip)
    else:
        output = m
    return output

from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import ELU

NORMAL_ACTIVATIONS = ['softmax', 'softplus', 'relu',
                      'tanh', 'sigmoid', 'hard_sigmoid', 'linear']


def denses(ip=None, input_shape=None, hiddens=None, is_dropout=True, activation='elu', name='denses', dropout_rate=0.5, **kwargs):
    """ sequential dense layers """
    if hiddens is None:
        hiddens = [128]
    n_hidden_layers = len(hiddens) 
    if ip is not None:
        input_shape = ip.get_shape()[1:].as_list()        
    m = Sequential(name=name)
    for i in range(n_hidden_layers):
        if i == 0:
            m.add(Dense(hiddens[i], activation=activation,
                        input_shape=input_shape, name=name + '/Dense_%d' % i))
        else:
            m.add(Dense(hiddens[i], activation=activation,
                        name=name + '/Dense_%d' % i))
        if is_dropout:
            m.add(Dropout(dropout_rate, name=name + '/Dropout_%d' % i))
    if ip is not None:
        output = m(ip)
    else:
        output = m
    return output


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

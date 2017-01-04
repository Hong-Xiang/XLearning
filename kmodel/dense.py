from keras.models import Model
from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import ELU

NORMAL_ACTIVATIONS = ['softmax', 'softplus', 'relu',
                      'tanh', 'sigmoid', 'hard_sigmoid', 'linear']


def dense(input_, hiddens, is_dropout, activation='elu', name='seq_dense', p=0.5, **kwargs):
    """ sequential dense layers """
    is_normal_activation = activation in NORMAL_ACTIVATIONS
    if not is_normal_activation:
        if activation == 'elu':
            activ = ELU

    x = input_
    n_hidden_layers = len(hiddens)    
    for i in range(n_hidden_layers):        
        if is_normal_activation:
            x = Dense(hiddens[i], activation=activation)(x)
        else:
            x = Dense(hiddens[i])(x)
            x = activ(**kwargs)(x)
        if is_dropout:
            x = Dropout(p)(x)    
    return x

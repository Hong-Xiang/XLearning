""" Core custom layers """
from keras.layers import Lambda
import keras.backend as K

def sub_fun(tensors):
    """ sub function for lambda """
    x, y = tensors
    return x - y

def sub(x, y, scope=''):
    """ functional call for sub layer """
    sub_layer = Lambda(sub_fun, name=scope+'sub')
    return sub_layer([x, y])
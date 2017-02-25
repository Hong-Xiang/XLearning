import tensorflow as tf
import numpy as np
from keras.layers import Convolution2D, BatchNormalization, Activation
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
        x = Convolution2D(nb_filter, nb_row, nb_col, border_mode=border_mode, name='convolution')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
    return x




import tensorflow as tf
import numpy as np
import xlearn.utils.general as utg

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


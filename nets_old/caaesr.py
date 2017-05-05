import tensorflow as tf
import tf.contrib.layers as ly
from .base import Net
from ..utils.general import with_config

def sr_1501_00092(inputs):
    """ simplest implementation of super resolution """
    with tf.name_scope('super_resolution'):
        c1 = ly.conv2d(inputs, 64, 9, padding='SAME', activation_fn=tf.nn.relu)
        c2 = ly.conv2d(c1, 32, 1, padding='SAME', activation_fn=tf.nn.relu)
        c3 = ly.conv2d(c2, 1, 5, padding='SAME')
    return c3



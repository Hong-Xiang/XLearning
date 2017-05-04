import tensorflow as tf
from ..utils.general import with_config
from .base import Net
from tensorflow.contrib import slim


class Cali0(Net):
    @with_config
    def __init__(self,

                 **kwargs):
        Net.__init__(self, **kwargs)
        self.params['name'] = "Cali0"

    def _set_model(self):
        self.input['data'] = tf.placeholder(
            dtype=tf.float32, shape=[None, 1, 10, 10], name='input')
        self.label['label'] = tf.placeholder(
            dtype=tf.float32, shape=[None, 2], name='label')
        h = self.input['data']
        with slim.arg_scope([slim.conv2d], padding='same', activation_fn=tf.nn.relu, data_format='NCHW'):
            h = slim.conv2d(h, 2**5, 3)
            h1 = h
            h = slim.conv2d(h, 2**5, 1)
            h2 = h
            h = slim.conv2d(h, 2**6, 1, 2)
            h3 = h
            h = slim.conv2d(h, 2**6, 1)
            h4 = h
            h = slim.conv2d(h, 2**7, 1, 2)
            h5 = h
            h = slim.conv2d(h, 2**7, 3)
            h = slim.conv2d(h, 2**7, 1)
            he = h
        h = slim.flatten(h)
        reps = [h]
        h = self.input['data']
        h = slim.flatten(h)
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
            filters = [2**d for d in range(7, 12)]
            h = slim.stack(h, slim.fully_connected, filters)
        reps.append(h)
        h = tf.concat(reps, axis=-1)
        # h = tf.layers.dense(h, 2**12)
        # h = tf.layers.dense(h, 2**13)
        h = slim.fully_connected(h, 2**12, activation_fn=tf.nn.relu)
        h = slim.fully_connected(h, 2**13, activation_fn=tf.nn.relu)
        h = tf.nn.dropout(h, self.kp)
        pos_pred = h = slim.fully_connected(h, 2, activation_fn=tf.nn.relu)
        self.output['pos_pred'] = pos_pred
        error = self.label['label'] - self.output['pos_pred']
        l = tf.square(error)
        l = tf.reduce_sum(l, axis=1)
        l = tf.sqrt(l)
        l = tf.reduce_mean(l)
        self.loss['loss'] = l
        tf.summary.scalar('loss', self.loss['loss'])
        self.train_op['main'] = tf.train.AdamOptimizer(
            self.lr['default']).minimize(self.loss['loss'], global_step=self.gs)
        self.summary_op['all'] = tf.summary.merge_all()

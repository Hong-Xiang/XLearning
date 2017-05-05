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
        h = slim.flatten(h)
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
            filters = [128, 256, 512, 1024, 2048, 4096, 8192]
            h = slim.stack(h, slim.fully_connected, filters)
        h = tf.nn.dropout(h, self.kp)
        pos_pred = slim.fully_connected(h, 2, activation_fn=None)
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


class Cali1(Net):
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
        with slim.arg_scope([slim.conv2d], padding='same', activation_fn=tf.nn.elu, data_format='NCHW'):
            h = slim.conv2d(h, 32, 3)
            h = slim.conv2d(h, 32, 1)
            h = slim.conv2d(h, 64, 1, 2)
            h = slim.conv2d(h, 64, 1)
            h = slim.conv2d(h, 128, 1, 2)
            h = slim.conv2d(h, 128, 3)
            h = slim.conv2d(h, 128, 1)
        h = slim.flatten(h)
        # h = slim.fully_connected(h, 2048, activation_fn=tf.nn.elu)
        reps = [h]
        h = self.input['data']
        h = slim.flatten(h)
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.elu):
            filters = [128, 256, 512, 1024, 2048]
            h = slim.stack(h, slim.fully_connected, filters)
        reps.append(h)

        h = tf.concat(reps, axis=-1)
        h = slim.fully_connected(h, 4096, activation_fn=tf.nn.elu)
        h = slim.fully_connected(h, 8192, activation_fn=tf.nn.elu)
        h = tf.nn.dropout(h, self.kp)
        pos_pred = slim.fully_connected(h, 2, activation_fn=None)
        self.output['pos_pred'] = pos_pred
        error = self.label['label'] - self.output['pos_pred']
        l = tf.square(error)
        l = tf.reduce_sum(l, axis=1)
        l = tf.sqrt(l)
        l = tf.reduce_mean(l)
        self.loss['loss'] = l
        tf.summary.scalar('loss', self.loss['loss'])
        self.train_op['main'] = tf.train.RMSPropOptimizer(
            self.lr['default']).minimize(self.loss['loss'], global_step=self.gs)
        self.summary_op['all'] = tf.summary.merge_all()

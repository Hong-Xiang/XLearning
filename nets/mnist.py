import tensorflow as tf
from ..utils.general import with_config
from .base import Net


class MNISTRecon0(Net):
    @with_config
    def __init__(self,
                 **kwargs):
        Net.__init__(self, **kwargs)
        self.params['name'] = "MNISTRecon0"

    def _set_model(self):
        self.input['data'] = tf.placeholder(
            dtype=tf.float32, shape=[None, 1, 28, 28], name='input')

        self.label['label'] = tf.placeholder(
            dtype=tf.int32, shape=[None, 1], name='label')
        h = self.input['data']
        h = tf.layers.conv2d(h, 32, 5, activation=tf.nn.relu,
                             data_format='channels_first', padding='same')
        h = tf.layers.conv2d(h, 64, 1, strides=(
            2, 2), activation=tf.nn.relu, data_format='channels_first', padding='valid')
        h = tf.layers.conv2d(h, 64, 5, activation=tf.nn.relu,
                             data_format='channels_first', padding='same')
        h = tf.layers.conv2d(h, 64, 1, strides=(
            2, 2), activation=tf.nn.relu, data_format='channels_first', padding='valid')
        h = tf.reshape(h, [-1, 7 * 7 * 64])
        h = tf.layers.dense(h, 1024, activation=tf.nn.relu)
        h = tf.nn.dropout(h, self.kp)
        y_pred = tf.layers.dense(h, 10)
        self.output['y_pred'] = y_pred
        self.output['dig_pred'] = tf.argmax(y_pred, 1)
        y_label = self.label['label']
        y_one_hot = tf.one_hot(y_label, 10)
        y_one_hot = tf.reshape(y_one_hot, [-1, 10])
        self.loss['loss'] = tf.losses.sigmoid_cross_entropy(
            multi_class_labels=y_one_hot, logits=y_pred)
        tf.summary.scalar('loss', self.loss['loss'])
        self.train_op['main'] = tf.train.AdamOptimizer(
            self.lr['default']).minimize(self.loss['loss'], global_step=self.gs)
        _, accuracy = tf.metrics.accuracy(
            self.label['label'], self.output['dig_pred'])
        self.metric['acc'] = accuracy

        tf.summary.scalar('acc', self.metric['acc'])
        self.summary_op['all'] = tf.summary.merge_all()

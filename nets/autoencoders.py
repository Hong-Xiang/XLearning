import tensorflow as tf
import tensorflow.contrib.slim as slim

from ..utils.general import with_config
from .base import Net


def AutoEncoderImage(Net):
    @with_config
    def __init__(self,
                 input_shape,
                 base_filter,
                 crop_size,
                 **kwargs):
        Net.__init__(self, **kwargs)
        self.params['input_shape'] = input_shape
        self.params['base_filter'] = base_filter
        self.params['crop_size'] = crop_size

    def _set_model(self):
        with tf.name_scope('data'):
            x_in = tf.placeholder(
                dtype=tf.float32, shape=self.input_shape['input'], name='data')
            self.add_node(x_in, 'data')
        h = data
        conv_cfgs = {
            'activation': tf.nn.elu,
            'padding': 'same',
            'data_format': 'channel_first'
        }
        f = self.params['base_filters']
        h = tf.layers.conv2d(h, f, 3, **conv_cfgs)
        h = tf.layers.conv2d(h, f, 3, **conv_cfgs)
        h = tf.layers.conv2d(h, 2 * f, 2, 2, **conv_cfgs)
        h = tf.layers.conv2d(h, 2 * f, 3, **conv_cfgs)
        h = tf.layers.conv2d(h, 2 * f, 3, **conv_cfgs)
        h = tf.layers.conv2d(h, 3 * f, 2, 2, **conv_cfgs)
        h = tf.layers.conv2d(h, 3 * f, 3, **conv_cfgs)
        h = tf.layers.conv2d(h, 3 * f, 3, **conv_cfgs)
        h = tf.layers.conv2d(h, 4 * f, 2, 2, **conv_cfgs)
        h = tf.layers.conv2d(h, 4 * f, 3, **conv_cfgs)
        h = tf.layers.conv2d(h, 4 * f, 3, **conv_cfgs)
        h = slim.flatten()
        shape_last = h.shape().to_list()
        dim_latent = int(np.prod(dim_latent[1:]))
        self.latent = tf.layers.dense(h, dim_latent)
        h = tf.reshape(self.latent, [-1, shape_last[1], shape_last[2], shape_last[3]])
        


        for f in reversed(self.params['filters']):
            h = tf.layers.conv2d(h, f, 3, **conv_cfgs)
            h = tf.layers.conv2d(h, 1, 3, **conv_cfgs)
            h = tf.contrib.keras.layers.Upsampling2D([2, 2])(h)
            h = tf.layers.conv2d(h, f // 2, 3, **conv_cfgs)
        pred = tf.layers.conv2d(h, 1, 3, padding='same',
                                data_format='channels_first')
        self.add_node(pred, 'pred')
        loss = tf.losses.mean_squared_error(pred, x_in)
        self.loss['loss'] = loss
        self.train_op['default'] = tf.train.AdamOptimizer(
            self.lr['default']).minimize(loss, global_step=self.gs)

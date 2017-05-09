import tensorflow as tf
from ..utils.general import with_config
from .base import Net
from ..models.image import conv2d, upsampling2d
from tensorflow.contrib import slim

class SRNetBase(Net):
    @with_config
    def __init__(self,
                 nb_down_sample,
                 is_down_sample_0=True,
                 is_down_sample_1=True,
                 upsampling_method='interp',
                 crop_size=None,
                 **kwargs):
        Net.__init__(self, **kwargs)
        self.params['is_down_sample_0'] = is_down_sample_0
        self.params['is_down_sample_1'] = is_down_sample_1
        self.params['nb_down_sample'] = nb_down_sample
        self.params['upsampling_method'] = upsampling_method
        self.params['crop_size'] = crop_size
        

class SRNet0(Net):
    @with_config
    def __init__(self,
                 low_shape,
                 filters,                 
                 high_shape,
                 data_format='channels_first',
                 optimizer_name='Adam',
                 **kwargs):
        Net.__init__(self, **kwargs)
        self.params['low_shape'] = low_shape
        self.params['name'] = "SRNetBase"
        self.params['filters'] = filters
        self.params['optimizer_name'] = optimizer_name
        self.params['data_format'] = data_format

    def _set_model(self):
        self.node['low_resolution'] = tf.placeholder(
            dtype=tf.float32, shape=self.params['low_shape'], name='low_resolution')
        self.node['high_resolution'] = tf.placeholder(
            dtype=tf.float32, shape=self.params['high_shape'], name='high_resolution')
        h = self.node['low_resolution']
        height = h.shape.as_list()[1]
        down_sample_ratio = [1, 1]
        if self.params['is_down_sample_0']:
            down_sample_ratio[0] *= 2
        h = upsampling2d(h, size=)
        h = tf.layers.conv2d(h, 64, 9, padding='same', data_format='channels_first', name='conv0')
        h = tf.layers.conv2d(h, 32, 1, padding='valid')
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
            if self.params['filters'] is None:
                filters = [128, 256, 512, 1024, 2048, 4096, 8192]
            else:
                filters = self.params['filters']
                # filters = [128, 256, 512, 1024, 2048, 4096, 8192]
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
        if self.params['optimizer_name'] == 'Adam':
            optim = tf.train.AdamOptimizer(self.lr['default'])
        else:
            optim = tf.train.RMSPropOptimizer(self.lr['default'])
        self.train_op['main'] = optim.minimize(
            self.loss['loss'], global_step=self.gs)
        self.summary_op['all'] = tf.summary.merge_all()

from tensorflow.python.ops import variable_scope as vs
vs.get_variable()
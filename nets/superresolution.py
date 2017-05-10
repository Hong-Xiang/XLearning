import tensorflow as tf
from ..utils.general import with_config
from .base import Net
from ..models.image import conv2d, upsampling2d
from tensorflow.contrib import slim


class SRNetBase(Net):
    @with_config
    def __init__(self,
                 batch_size,
                 low_shape,
                 nb_down_sample,
                 is_down_sample_0=True,
                 is_down_sample_1=True,
                 upsampling_method='interp',
                 high_shape=None,
                 crop_size=None,
                 **kwargs):
        Net.__init__(self, **kwargs)
        self.params['is_down_sample_0'] = is_down_sample_0
        self.params['is_down_sample_1'] = is_down_sample_1
        self.params['nb_down_sample'] = nb_down_sample
        self.params['upsampling_method'] = upsampling_method
        self.params['crop_size'] = crop_size
        self.params['batch_size'] = batch_size
        down_sample_ratio = [1, 1]
        if self.params['is_down_sample_0']:
            down_sample_ratio[0] *= 2
        if self.params['is_down_sample_1']:
            down_sample_ratio[1] *= 2
        down_sample_ratio[0] = down_sample_ratio[0]**self.params['nb_down_sample']
        down_sample_ratio[1] = down_sample_ratio[1]**self.params['nb_down_sample']
        self.params['down_sample_ratio'] = down_sample_ratio
        self.params['low_shape'] = low_shape
        if high_shape is None:
            high_shape = list(self.params['low_shape'])
            high_shape[0] *= self.params['down_sample_ratio'][0]
            high_shape[1] *= self.params['down_sample_ratio'][1]
        else:
            self.params['high_shape'] = high_shape


class SRNet0(SRNetBase):
    @with_config
    def __init__(self,
                 low_shape,
                 filters,
                 high_shape,
                 **kwargs):
        Net.__init__(self, **kwargs)
        self.params['name'] = "SRNetBase"
        self.params['filters'] = filters

    def _set_model(self):
        low_shape = [self.params['batch_size']] + \
            list(self.params['low_shape']) + [1]
        high_shape = [self.params['batch_size']] + \
            list(self.params['high_shape']) + [1]
        low_res = tf.placeholder(
            dtype=tf.float32, shape=self.params['low_shape'], name='low_resolution')
        self.node['low_resolution'] = low_res
        high_res = tf.placeholder(
            dtype=tf.float32, shape=self.params['high_shape'], name='high_resolution')
        self.node['high_resolution'] = high_res
        h = self.node['low_resolution']
        h = upsampling2d(h, size=self.params['down_sample_ratio'])
        tf.summary.image('interp', h)
        res_ref = high_res - h
        tf.summary.image('res_ref', res_ref)
        h = tf.layers.conv2d(h, 64, 9, padding='same',
                             name='conv0', activation=tf.nn.elu)
        h = tf.layers.conv2d(h, 32, 1, padding='same',
                             name='conv1', activation=tf.nn.elu)
        h = tf.layers.conv2d(h, 32, 1, padding='same', name='conv2')
        tf.summary.image('res_inf', h)
        sr_res = h + low_res
        tf.summary.image('sr_res', sr_res)
        self.node['super_resolution'] = sr_res
        self.loss['loss'] = tf.losses.mean_squared_error(high_res, sr_res)
        tf.summary.scalar('loss', self.loss['loss'])
        optim = tf.train.RMSPropOptimizer(self.lr['default'])
        self.train_op['main'] = optim.minimize(
            self.loss['loss'], global_step=self.gs)
        self.summary_op['all'] = tf.summary.merge_all()

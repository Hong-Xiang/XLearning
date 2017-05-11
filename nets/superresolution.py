import tensorflow as tf
from ..utils.general import with_config
from .base import Net
from ..models.image import conv2d, upsampling2d
from tensorflow.contrib import slim
import numpy as np

class SRNetBase(Net):
    @with_config
    def __init__(self,
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
        
        down_sample_ratio = [1, 1]
        if self.params['is_down_sample_0']:
            down_sample_ratio[0] *= 2
        if self.params['is_down_sample_1']:
            down_sample_ratio[1] *= 2
        down_sample_ratio[0] = down_sample_ratio[0]**self.params['nb_down_sample']
        down_sample_ratio[1] = down_sample_ratio[1]**self.params['nb_down_sample']
        self.params['down_sample_ratio'] = down_sample_ratio
        self.params['low_shape'] = [self.params['batch_size']] + list(low_shape) + [1]
        if high_shape is None:
            high_shape = list(self.params['low_shape'])
            high_shape[1] *= self.params['down_sample_ratio'][0]
            high_shape[2] *= self.params['down_sample_ratio'][1] ** self.params['nb_down_sample']
            self.params['high_shape'] = high_shape
        else:
            self.params['high_shape'] = [self.params['batch_size']] + list(high_shape) + [1]


class SRNet0(SRNetBase):
    """Dong C, Loy CC, He K, Tang X. Image Super-Resolution Using Deep Convolutional Networks. IEEE Trans Pattern Anal Mach Intell. 2016;38(2):295-307. doi:10.1109/TPAMI.2015.2439281."""
    @with_config
    def __init__(self,                 
                 **kwargs):
        SRNetBase.__init__(self, **kwargs)
        self.params['name'] = "SRNet0"

    def _set_model(self):        
        low_res = tf.placeholder(
            dtype=tf.float32, shape=self.params['low_shape'], name='low_resolution')
        self.node['low_resolution'] = low_res
        self.input['data'] = low_res
        high_res = tf.placeholder(
            dtype=tf.float32, shape=self.params['high_shape'], name='high_resolution')
        self.node['high_resolution'] = high_res
        self.label['label'] = high_res
        h = self.node['low_resolution']        
        itp = upsampling2d(h, size=self.params['down_sample_ratio'])        
        tf.summary.image('interp', h)
        res_ref = high_res - itp
        tf.summary.image('res_ref', res_ref)
        h = tf.layers.conv2d(itp, 64, 9, padding='same',
                             name='conv0', activation=tf.nn.elu)
        h = tf.layers.conv2d(h, 32, 1, padding='same',
                             name='conv1', activation=tf.nn.elu)
        h = tf.layers.conv2d(h, 1, 5, padding='same', name='conv2')
        tf.summary.image('res_inf', h)
        sr_res = h + itp
        tf.summary.image('sr_res', sr_res)
        self.node['super_resolution'] = sr_res
        with tf.name_scope('loss'):
            self.loss['loss'] = tf.losses.mean_squared_error(high_res, sr_res)/self.params['batch_size']
        tf.summary.scalar('loss', self.loss['loss'])
        optim = tf.train.RMSPropOptimizer(self.lr['default'])
        self.train_op['main'] = optim.minimize(
            self.loss['loss'], global_step=self.gs)
        self.summary_op['all'] = tf.summary.merge_all()

class SRNet1(SRNetBase):
    @with_config
    def __init__(self,                 
                 **kwargs):
        SRNetBase.__init__(self, **kwargs)
        self.params['name'] = "SRNet1"


    def _set_model(self):        
        low_res = tf.placeholder(
            dtype=tf.float32, shape=self.params['low_shape'], name='low_resolution')
        self.node['low_resolution'] = low_res
        self.input['data'] = low_res
        high_res = tf.placeholder(
            dtype=tf.float32, shape=self.params['high_shape'], name='high_resolution')
        self.node['high_resolution'] = high_res
        self.label['label'] = high_res
        h = self.node['low_resolution']
        self.debug_tensor['low_reso'] = h
        self.debug_tensor['high_reso'] = high_res
        itp = upsampling2d(h, size=self.params['down_sample_ratio'])
        self.debug_tensor['ipt'] = itp
        tf.summary.image('interp', h)
        res_ref = high_res - itp
        tf.summary.image('res_ref', res_ref)
        h = tf.layers.conv2d(itp, 64, 3, padding='same',
                                name='conv_stem', activation=tf.nn.elu)        
        for i in range(20):
            h = tf.layers.conv2d(h, 64, 3, padding='same',
                                name='conv_%d'%i, activation=tf.nn.elu)        
        h = tf.layers.conv2d(h, 1, 5, padding='same', name='conv_end')
        tf.summary.image('res_inf', h)
        self.debug_tensor['inf'] = h
        sr_res = h + itp
        self.debug_tensor['inf_full'] = sr_res
        tf.summary.image('sr_res', sr_res)
        self.node['super_resolution'] = sr_res
        with tf.name_scope('loss'):
            self.loss['loss'] = tf.losses.mean_squared_error(high_res, sr_res)/self.params['batch_size']
        tf.summary.scalar('loss', self.loss['loss'])
        optim = tf.train.RMSPropOptimizer(self.lr['default'])
        self.train_op['main'] = optim.minimize(
            self.loss['loss'], global_step=self.gs)
        self.summary_op['all'] = tf.summary.merge_all()

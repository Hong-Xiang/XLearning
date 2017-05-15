import tensorflow as tf
from ..utils.general import with_config
from .base import Net
from ..models.image import conv2d, upsampling2d
from tensorflow.contrib import slim
import numpy as np
from ..utils.general import analysis_device

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
        self.params['low_shape'] = [
            self.params['batch_size']] + list(low_shape) + [1]
        if high_shape is None:
            high_shape = list(self.params['low_shape'])
            high_shape[1] *= self.params['down_sample_ratio'][0]
            high_shape[2] *= self.params['down_sample_ratio'][1] ** self.params['nb_down_sample']
            self.params['high_shape'] = high_shape
        else:
            self.params['high_shape'] = [
                self.params['batch_size']] + list(high_shape) + [1]
        self.params.update_short_cut()


class SRNet0(SRNetBase):
    """Dong C, Loy CC, He K, Tang X. Image Super-Resolution Using Deep Convolutional Networks. IEEE Trans Pattern Anal Mach Intell. 2016;38(2):295-307. doi:10.1109/TPAMI.2015.2439281."""
    @with_config
    def __init__(self,
                 **kwargs):
        SRNetBase.__init__(self, **kwargs)
        self.params['name'] = "SRNet0"
        self.params.update_short_cut()

    def _set_model(self):
        low_res = tf.placeholder(
            dtype=tf.float32, shape=self.params['low_shape'], name='low_resolution')
        self.add_node(low_res, 'data')
        high_res = tf.placeholder(
            dtype=tf.float32, shape=self.params['high_shape'], name='high_resolution')
        self.add_node(high_res, 'label')
        h = self.node['low_resolution']
        itp = upsampling2d(
            h, size=self.params['down_sample_ratio'], method='bilinear')
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
            self.loss['loss'] = tf.losses.mean_squared_error(
                high_res, sr_res) / self.params['batch_size']
        tf.summary.scalar('loss', self.loss['loss'])
        optim = tf.train.RMSPropOptimizer(self.lr)
        self.train_op['main'] = optim.minimize(
            self.loss['loss'], global_step=self.gs)
        self.summary_op['all'] = tf.summary.merge_all()


class SRNet1(SRNetBase):
    @with_config
    def __init__(self,
                 filters=64,
                 depths=20,
                 train_verbose=1,
                 **kwargs):
        SRNetBase.__init__(self, **kwargs)
        self.params['name'] = "SRNet1"
        self.params['filters'] = filters
        self.params['depths'] = depths
        self.params['train_verbose'] = train_verbose
        self.params.update_short_cut()
    
    def super_resolution(self, low_res, high_res, with_summary=False, reuse=None, name=None):
        with tf.name_scope(name):
            h = low_res            
            interp = upsampling2d(h, size=self.params['down_sample_ratio'])                        
            res_ref = high_res - interp       
            err_itp = tf.abs(res_ref)
            
            h = tf.layers.conv2d(interp, 64, 5, padding='same',
                                name='conv_stem', activation=tf.nn.elu, reuse=reuse)
            for i in range(self.params['depths']):
                h = tf.layers.conv2d(
                    h, self.params['filters'], 3, padding='same', name='conv_%d' % i, activation=tf.nn.elu, reuse=reuse)
                if with_summary:
                    tf.summary.histogram('activ_%d' % i, h)
            res_inf = tf.layers.conv2d(h, 1, 5, padding='same', name='conv_end', reuse=reuse)            
            sr_inf = interp + res_inf
            err_inf = tf.abs(high_res - sr_inf)
            
            patch_size = self.p.high_shape[1] * self.p.high_shape[2]
            loss = tf.losses.mean_squared_error(high_res, sr_inf) / patch_size
            grad = self.optimizers['train'].compute_gradients(loss)
            
            if with_summary:
                tf.summary.image('low_res', low_res)
                tf.summary.image('high_res', high_res)
                tf.summary.image('interp', interp)
                tf.summary.image('sr_inf', sr_inf)
                tf.summary.image('res_ref', res_ref)
                tf.summary.image('res_inf', res_inf)
                tf.summary.image('err_itp', err_itp)
                tf.summary.image('err_inf', err_inf)
                tf.summary.scalar('loss', loss)            
            return sr_inf, loss, grad
                

    def _set_model(self):
        bs_gpu = self.p.batch_size // self.p.nb_gpus
        sliced_low_res = []
        sliced_high_res = []
        with tf.device('/cpu:0'):
            with tf.name_scope('low_resolution'):
                low_res = tf.placeholder(
                    dtype=tf.float32, shape=self.params['low_shape'], name='low_resolution')
                self.add_node(low_res, 'data')
                gpu_shape = low_res.shape.as_list()
                gpu_shape[0] = bs_gpu
                for i in range(self.p.nb_gpus):
                    with tf.name_scope('device_%d'%i):
                        sliced_low_res.append(tf.slice(low_res, [i*bs_gpu, 0, 0, 0], gpu_shape))                
            with tf.name_scope('high_resolution'):
                high_res = tf.placeholder(                
                    dtype=tf.float32, shape=self.params['high_shape'], name='high_resolution')
                self.add_node(high_res, 'label')
                gpu_shape = high_res.shape.as_list()
                gpu_shape[0] = bs_gpu
                for i in range(self.p.nb_gpus):
                    with tf.name_scope('device_%d'%i):
                        sliced_high_res.append(tf.slice(high_res, [i*bs_gpu, 0, 0, 0], gpu_shape))                
            self.optimizers['train'] = tf.train.RMSPropOptimizer(self.lr['train'])
            self.super_resolution(sliced_low_res[0], sliced_high_res[0], with_summary=False, reuse=None, name='cpu_tower')
        
        sr_infs = []
        losses = []
        grads = []
        for i in range(self.p.nb_gpus):
            device = '/gpu:%d'%i
            with tf.device(device):
                sr_inf, loss, grad = self.super_resolution(sliced_low_res[i], sliced_high_res[i], with_summary=False, reuse=True, name='gpu_%d'%i)
            sr_infs.append(sr_inf)
            losses.append(loss)
            grads.append(grad)

        with tf.name_scope('loss'):
            self.losses['train'] = tf.add_n(losses)
        
        with tf.name_scope('infer'):
            sr_inf = tf.concat(sr_infs, axis=0)
            self.add_node(sr_inf, 'inference')
        
        with tf.name_scope('cpu_summary'):
            interp = upsampling2d(low_res, size=self.params['down_sample_ratio'])                        
            res_ref = high_res - interp
            res_inf = high_res - sr_inf      
            err_itp = tf.abs(res_ref)            
            err_inf = tf.abs(res_inf)
            l2_err_itp = tf.reduce_sum(tf.square(err_itp), axis=[1, 2, 3])
            l2_err_inf = tf.reduce_sum(tf.square(err_inf), axis=[1, 2, 3])

            l2_inf = tf.reduce_mean(l2_err_inf)
            l2_itp = tf.reduce_mean(l2_err_itp)
            ratio = tf.reduce_mean(l2_err_inf/(l2_err_itp+1e-3))
            tf.summary.image('low_res', low_res)
            tf.summary.image('high_res', high_res)
            tf.summary.image('interp', interp)
            tf.summary.image('sr_inf', sr_inf)
            tf.summary.image('res_ref', res_ref)
            tf.summary.image('res_inf', res_inf)
            tf.summary.image('err_itp', err_itp)
            tf.summary.image('err_inf', err_inf)
            tf.summary.scalar('loss', self.losses['train'])     
            tf.summary.scalar('ratio', ratio)
            tf.summary.scalar('l2_inf', l2_inf)
            tf.summary.scalar('l2_itp', l2_itp)

        train_step = self.train_step(grads, self.optimizers['train'], summary_verbose=self.p.train_verbose)
        self.train_steps['train'] = train_step
        self.summary_ops['all'] = tf.summary.merge_all()

        self.feed_dict['train'] = ['data', 'label']
        self.feed_dict['predict'] = ['data']
        self.feed_dict['summary'] = ['data', 'label']

        self.run_op['train'] = {'train_step': self.train_steps['train'],
                                'loss': self.losses['train'], 'global_step': self.gs}
        self.run_op['predict'] = {'infernce': sr_inf}
        self.run_op['summary'] = {'summary_all': self.summary_ops['all']}
    
    def _set_train(self):
        pass
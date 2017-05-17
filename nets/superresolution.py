import tensorflow as tf
from ..utils.general import with_config
from .base import Net
from ..models.image import conv2d, upsampling2d, align_by_crop, residual2
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
                 train_verbose=0,
                 is_bn=True,
                 is_res=True,
                 **kwargs):
        SRNetBase.__init__(self, **kwargs)
        self.params['name'] = "SRNet1"
        self.params['filters'] = filters
        self.params['depths'] = depths
        self.params['train_verbose'] = train_verbose
        self.params['is_bn'] = is_bn
        self.params['is_res'] = is_res
        self.params.update_short_cut()
    
    def super_resolution(self, low_res, high_res, with_summary=False, reuse=None, name=None):
        with tf.name_scope(name):
            h = low_res            
            interp = upsampling2d(h, size=self.params['down_sample_ratio'])                        
 
            
            h = tf.layers.conv2d(interp, 64, 5, padding='same',
                                name='conv_stem', activation=tf.nn.elu, reuse=reuse)
            for i in range(self.params['depths']):
                hpre = h
                h = tf.layers.conv2d(
                    h, self.params['filters'], 3, padding='same', name='conv_%d' % i, reuse=reuse)
                if self.p.is_bn:
                    h = tf.layers.batch_normalization(h, training=self.training, reuse=reuse, name='bn_%d'%i, scale=False)
                if self.p.is_res:
                    h = 0.2 * h + hpre
                h = tf.nn.relu(h)
                
                if with_summary:
                    tf.summary.histogram('activ_%d' % i, h)
            res_inf = tf.layers.conv2d(h, 1, 5, padding='same', name='conv_end', reuse=reuse)                      
                             
            sr_inf = interp + res_inf
                        
           

            res_ref = high_res - interp  
            err_itp = tf.abs(res_ref)  
            err_inf = tf.abs(high_res - sr_inf)
            # patch_size = self.p.high_shape[1] * self.p.high_shape[2]
            loss = tf.losses.mean_squared_error(high_res, sr_inf)
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
            self.optimizers['train'] = tf.train.AdamOptimizer(self.lr['train'])
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
        tf.summary.scalar('lr/train', self.lr['train'])
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

        self.feed_dict['train'] = ['data', 'label', 'training']
        self.feed_dict['predict'] = ['data', 'training']
        self.feed_dict['summary'] = ['data', 'label', 'training']

        self.run_op['train'] = {'train_step': self.train_steps['train'],
                                'loss': self.losses['train'], 'global_step': self.gs}
        self.run_op['predict'] = {'infernce': sr_inf}
        self.run_op['summary'] = {'summary_all': self.summary_ops['all']}
    
    def _set_train(self):
        pass


class SRNet2(SRNetBase):
    @with_config
    def __init__(self,
                 filters=64,
                 depths=2,
                 hidden_num=8,
                 train_verbose=0,
                 **kwargs):
        SRNetBase.__init__(self, **kwargs)
        self.params['name'] = "SRNet1"
        self.params['filters'] = filters
        self.params['depths'] = depths
        self.params['train_verbose'] = train_verbose
        self.params['hidden_num'] = hidden_num
        self.params.update_short_cut()
    
    def gen(self, z, low_res, reuse=None):        
        hidden_num = self.p.hidden_num
        filters = self.p.filters        
        with tf.variable_scope('gen', reuse=reuse) as vs:
            with tf.name_scope('z1d2img8x8'):
                num_output = int(np.prod([8, 8, self.p.filters]))
                h = tf.layers.dense(z, num_output, activation_fn=None, name='dense')
                h = tf.reshape(x, shape=[-1, 8, 8, filters], name='reshape_latent')
            with tf.name_scope('img8x8'):
                h = residual2(h, filters, name='res_0')
                h = residual2(h, filters, name='res_1')
            h = tf.image.resize_nearest_neighbor(h, size=[16, 8])
            h = residual2(h, filters, name='res_2')
            h = residual2(h, filters, name='res_3')
            h = tf.image.resize_nearest_neighbor(h, size=[32, 8])
            h = residual2(h, filters, name='res_4')
            h = residual2(h, filters, name='res_5')
            h = tf.image.resize_nearest_neighbor(h, size=[64, 8])
            h = [h, low_res]
            h = tf.concat(h, axis=-1)
            h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_6', activation=tf.nn.elu)
            h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_7', activation=tf.nn.elu)
            h = tf.image.resize_nearest_neighbor(h, size=[64, 16])
            h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_8', activation=tf.nn.elu)
            h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_9', activation=tf.nn.elu)
            out = tf.layers.conv2d(h, filters, 1, padding='same', name='conv_10')
        variables = tf.contrib.framework.get_variables(vs)
        return out, variables

    def dis(self, x, low_res, reuse=None):        
        hidden_num = self.p.hidden_num
        filters = self.p.filters
        with tf.name_scope('dis'):
            with tf.name_scope('enc'):
                with tf.variable_scope('enc', reuse=reuse) as vs0:
                    h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_0', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_1', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters, 3, (1, 2), padding='same', name='conv_2', activation=tf.nn.elu)
                    h = [h, low_res]
                    h = tf.concat(h, axis=-1)
                    h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_3', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_4', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters*2, 3, (2, 1), padding='same', name='conv_5', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters*2, 3, padding='same', name='conv_6', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters*2, 3, padding='same', name='conv_7', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters*3, 3, 2, padding='same', name='conv_8', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters*3, 3, padding='same', name='conv_9', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters*3, 3, padding='same', name='conv_10', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters*3, 3, 2, padding='same', name='conv_11', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters*4, 3, padding='same', name='conv_9', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters*4, 3, padding='same', name='conv_10', activation=tf.nn.elu)
                    h = tf.contrib.slim.flatten(h)            
                    embed = tf.layers.dense(h, hidden_num, name='dense')
                variables = tf.contrib.framework.get_variables(vs0)
            with tf.name_scope('dec'):
                with tf.variable_scope('dec', reuse=reuse) as vs1:                                                       
                    h = tf.layers.dense(embed, num_output, activation_fn=None, name='dense')
                    h = tf.reshape(x, shape=[-1, 8, 8, filters], name='reshape_latent')
                    h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_0', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_1', activation=tf.nn.elu)
                    h = tf.image.resize_nearest_neighbor(h, size=[16, 16])
                    h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_2', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_3', activation=tf.nn.elu)
                    h = tf.image.resize_nearest_neighbor(h, size=[32, 32])
                    h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_4', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_5', activation=tf.nn.elu)
                    h = tf.image.resize_nearest_neighbor(h, size=[64, 32])
                    h = [h, low_res]
                    h = tf.concat(h, axis=-1)
                    h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_6', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_7', activation=tf.nn.elu)
                    h = tf.image.resize_nearest_neighbor(h, size=[64, 64])
                    h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_8', activation=tf.nn.elu)
                    h = tf.layers.conv2d(h, filters, 3, padding='same', name='conv_9', activation=tf.nn.elu)
                    out = tf.layers.conv2d(h, filters, 1, padding='same', name='conv_10')        
                variables.append(tf.contrib.framework.get_variables(vs1))
        return out, variables

    def _set_model(self):        
        with tf.name_scope('low_resolution'):
            low_res = tf.placeholder(
                dtype=tf.float32, shape=self.params['low_shape'], name='low_resolution')
            self.add_node(low_res, 'data')
        with tf.name_scope('high_resolution'):
            high_res = tf.placeholder(                
                dtype=tf.float32, shape=self.params['high_shape'], name='high_resolution')
            self.add_node(high_res, 'label')
        self.optimizers['train'] = tf.train.RMSPropOptimizer(self.lr['train'])
        interp = upsampling2d(low_res, size=self.params['down_sample_ratio'], name='interp_low_res')
        h = tf.layers.conv2d(low_res, self.p.filters, 5, padding='same', activation=tf.nn.elu, name='conv_stem')
        for i in range(self.p.depths):
            h = tf.layers.conv2d(h, self.p.filters, 3, padding='same', activation=tf.nn.elu, name='conv_%d'%i)
        rep_int = upsampling2d(h, size=self.params['down_sample_ratio'], name='interp_reps')
        res_inf = tf.layers.conv2d(h, self.p.filters, 5, padding='same', activation=tf.nn.elu, name='conv_end'%i)
        sr_inf = res_inf + interp
        
        loss = tf.losses.mean_squared_error(high_res, sr_inf)

        self.losses['train'] = loss

        train_step = self.optimizers['train'].minimize(loss)





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


class SRNet3(SRNetBase):
    @with_config
    def __init__(self,
                 filters=64,
                 depths=20,
                 train_verbose=0,
                 is_bn=True,
                 is_res=True,
                 **kwargs):
        SRNetBase.__init__(self, **kwargs)
        self.params['name'] = "SRNet3"
        self.params['filters'] = filters
        self.params['depths'] = depths
        self.params['train_verbose'] = train_verbose
        self.params['is_bn'] = is_bn
        self.params['is_res'] = is_res
        self.params.update_short_cut()
    
    def super_resolution(self, low_res, high_res, with_summary=False, reuse=None, name=None):
        with tf.name_scope(name):
            h = low_res            
            interp = upsampling2d(h, size=self.params['down_sample_ratio'])                        

            
            h = tf.layers.conv2d(interp,  self.params['filters'], 5, padding='same',
                                name='conv_stem', activation=tf.nn.elu, reuse=reuse)
            for i in range(self.params['depths']):
                hpre = h
                h = tf.layers.conv2d(
                    h, self.params['filters'], 3, padding='same', name='conv_%d' % i, reuse=reuse)
                if self.p.is_bn:
                    h = tf.layers.batch_normalization(h, training=self.training, reuse=reuse, name='bn_%d'%i, scale=False)
                if self.p.is_res:
                    h = 0.2 * h + hpre
                h = tf.nn.elu(h)
                
                if with_summary:
                    tf.summary.histogram('activ_%d' % i, h)
            res_inf = tf.layers.conv2d(h, 1, 5, padding='same', name='conv_end', reuse=reuse)            
            sr_inf = interp + res_inf
            
            res_inf, sr_inf, interp = align_by_crop(high_res, [res_inf, sr_inf, interp])

            err_inf = tf.abs(high_res - sr_inf)
            res_ref = high_res - interp
            err_itp = tf.abs(res_ref)

            patch_size = self.p.high_shape[1] * self.p.high_shape[2]
            loss = tf.losses.mean_squared_error(high_res, sr_inf)
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
            return sr_inf, loss, grad, interp, high_res
                

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
                gpu_shape[1] -= 2*self.p.crop_size
                gpu_shape[2] -= 2*self.p.crop_size
                for i in range(self.p.nb_gpus):
                    with tf.name_scope('device_%d'%i):
                        sliced_high_res.append(tf.slice(high_res, [i*bs_gpu, self.p.crop_size, self.p.crop_size, 0], gpu_shape))                
            self.optimizers['train'] = tf.train.AdamOptimizer(self.lr['train'])
            self.super_resolution(sliced_low_res[0], sliced_high_res[0], with_summary=False, reuse=None, name='cpu_tower')
        
        sr_infs = []
        losses = []
        grads = []
        interps = []
        high_ress = []
        for i in range(self.p.nb_gpus):
            device = '/gpu:%d'%i
            with tf.device(device):
                sr_inf, loss, grad, interp, high_res = self.super_resolution(sliced_low_res[i], sliced_high_res[i], with_summary=False, reuse=True, name='gpu_%d'%i)
            sr_infs.append(sr_inf)
            losses.append(loss)
            grads.append(grad)
            interps.append(interp)
            high_ress.append(high_res)

        with tf.name_scope('loss'):
            self.losses['train'] = tf.add_n(losses)
        
        with tf.name_scope('infer'):
            sr_inf = tf.concat(sr_infs, axis=0)
            self.add_node(sr_inf, 'inference')
            interp = tf.concat(interps, axis=0)
            high_res = tf.concat(high_ress, axis=0)
        tf.summary.scalar('lr/train', self.lr['train'])
        with tf.name_scope('cpu_summary'):
            
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

        self.feed_dict['train'] = ['data', 'label', 'training']
        self.feed_dict['predict'] = ['data', 'training']
        self.feed_dict['summary'] = ['data', 'label', 'training']

        self.run_op['train'] = {'train_step': self.train_steps['train'],
                                'loss': self.losses['train'], 'global_step': self.gs}
        self.run_op['predict'] = {'inference': sr_inf, 'interp': interp}
        self.run_op['summary'] = {'summary_all': self.summary_ops['all']}
    
    def _set_train(self):
        pass



class SRNet4(SRNetBase):
    @with_config
    def __init__(self,
                 filters=64,
                 depths=20,
                 train_verbose=0,
                 is_bn=True,
                 is_res=True,
                 **kwargs):
        SRNetBase.__init__(self, **kwargs)
        self.params['name'] = "SRNet4"
        self.params['filters'] = filters
        self.params['depths'] = depths
        self.params['train_verbose'] = train_verbose
        self.params['is_bn'] = is_bn
        self.params['is_res'] = is_res
        self.params.update_short_cut()
    
    def super_resolution(self, img8x, img4x, img2x, img1x, with_summary=False, reuse=None, name=None):
        cid = 0
        with tf.name_scope(name):            
            with tf.name_scope('net8x4x'):
                h = tf.layers.conv2d(img8x,  self.params['filters'], 5, padding='same',
                                    name='conv_stem', activation=tf.nn.elu, reuse=reuse)            
                for i in range(self.params['depths']//3):
                    hpre = h
                    h = tf.layers.conv2d(
                        h, self.params['filters'], 3, padding='same', name='conv_%d' % cid, reuse=reuse)
                    cid += 1
                    if self.p.is_bn:
                        h = tf.layers.batch_normalization(h, training=self.training, reuse=reuse, name='bn_%d'%cid, scale=False)
                    if self.p.is_res:
                        h = 0.2 * h + hpre
                    h = tf.nn.elu(h)
                    
                    if with_summary:
                        tf.summary.histogram('activ_%d' % i, h)                        
                h = upsampling2d(h, size=[1, 2])
                res4x = tf.layers.conv2d(h, 1, 5, padding='same', name='conv_4x', reuse=reuse)
                itp4x = upsampling2d(img8x, size=[1, 2])
                inf4x = res4x + itp4x

            with tf.name_scope('net8x4x'):
                for i in range(self.params['depths']//3):
                    hpre = h
                    h = tf.layers.conv2d(
                        h, self.params['filters'], 3, padding='same', name='conv_%d' % cid, reuse=reuse)
                    
                    if self.p.is_bn:
                        h = tf.layers.batch_normalization(h, training=self.training, reuse=reuse, name='bn_%d'%cid, scale=False)
                    if self.p.is_res:
                        h = 0.2 * h + hpre
                    cid += 1
                    h = tf.nn.elu(h)
                    
                    if with_summary:
                        tf.summary.histogram('activ_%d' % i, h)
                itp2x = upsampling2d(itp4x, size=[1, 2])
                h = upsampling2d(h, size=[1, 2])            
                res2x = tf.layers.conv2d(h, 1, 5, padding='same', name='conv_2x', reuse=reuse)
                inf2x = res2x + itp2x

            with tf.name_scope('net8x4x'):
                for i in range(self.params['depths']//3):
                    hpre = h
                    h = tf.layers.conv2d(
                        h, self.params['filters'], 3, padding='same', name='conv_%d' % cid, reuse=reuse)
                    if self.p.is_bn:
                        h = tf.layers.batch_normalization(h, training=self.training, reuse=reuse, name='bn_%d'%cid, scale=False)
                    if self.p.is_res:
                        h = 0.2 * h + hpre
                    cid += 1
                    h = tf.nn.elu(h)
                    
                    if with_summary:
                        tf.summary.histogram('activ_%d' % i, h)                                    
                itp1x = upsampling2d(itp2x, size=[1, 2])
                h = upsampling2d(h, size=[1, 2])  
                res1x = tf.layers.conv2d(h, 1, 5, padding='same', name='conv_1x', reuse=reuse)
                inf1x = res1x + itp1x

            with tf.name_scope('crop'):
                shape4x = img4x.shape.as_list()
                shape2x = img2x.shape.as_list()
                shape1x = img1x.shape.as_list()
                shape4x[1] -= self.p.crop_size // 2
                shape4x[2] -= self.p.crop_size // 2
                shape2x[1] -= self.p.crop_size
                shape2x[2] -= self.p.crop_size
                shape1x[1] -= self.p.crop_size*2 
                shape1x[2] -= self.p.crop_size*2
                img4x = tf.slice(img4x, [0, self.p.crop_size//4, self.p.crop_size//4, 0], shape4x)
                img2x = tf.slice(img2x, [0, self.p.crop_size//2, self.p.crop_size//2, 0], shape2x)
                img1x = tf.slice(img1x, [0, self.p.crop_size, self.p.crop_size, 0], shape1x)
                
                inf4x, res4x, itp4x = align_by_crop(img4x, [inf4x, res4x, itp4x])
                inf2x, res2x, itp2x = align_by_crop(img2x, [inf2x, res2x, itp2x])
                inf1x, res1x, itp1x = align_by_crop(img1x, [inf1x, res1x, itp1x])
                # res_inf, sr_inf, interp = align_by_crop(img1x, [res_inf, sr_inf, interp])

            
            with tf.name_scope('loss'):
                loss4x = tf.losses.mean_squared_error(img4x, inf4x)
                loss2x = tf.losses.mean_squared_error(img2x, inf2x)
                loss1x = tf.losses.mean_squared_error(img1x, inf1x)
                loss = 0.1 * loss4x + 0.5 * loss2x + loss1x            
                grad = self.optimizers['train'].compute_gradients(loss)
            
            return inf1x, loss, grad, itp1x, img1x
                
    def add_data(self, nb_down, shape):
        sliced = []
        bs_gpu = self.p.batch_size // self.p.nb_gpus
        name = 'data%d'%nb_down
        with tf.name_scope(name):
            data = tf.placeholder(
                dtype=tf.float32, shape=shape, name=name)
            self.add_node(data, name)
            gpu_shape = data.shape.as_list()
            gpu_shape[0] = bs_gpu
            for i in range(self.p.nb_gpus):
                with tf.name_scope('device_%d'%i):
                    sliced.append(tf.slice(data, [i*bs_gpu, 0, 0, 0], gpu_shape))
        return sliced

    def _set_model(self):
        bs_gpu = self.p.batch_size // self.p.nb_gpus

        sliced_data0 = []
        sliced_data1 = []
        sliced_data2 = []
        sliced_data3 = []
        shape4x = self.params['low_shape']
        shape3x = list(shape4x)
        shape3x[2] *= 2
        shape2x = list(shape3x)
        shape2x[2] *= 2
        shape1x = list(shape2x)
        shape1x[2] *= 2
        
        with tf.device('/cpu:0'):
            sliced4x = self.add_data(3, shape4x)            
            sliced3x = self.add_data(2, shape3x)            
            sliced2x = self.add_data(1, shape2x)            
            sliced1x = self.add_data(0, shape1x)            

            self.optimizers['train'] = tf.train.AdamOptimizer(self.lr['train'])
            self.super_resolution(sliced4x[0], sliced3x[0], sliced2x[0], sliced1x[0], with_summary=False, reuse=None, name='cpu_tower')
        
        sr_infs = []
        losses = []
        grads = []
        interps = []
        high_ress = []
        for i in range(self.p.nb_gpus):
            device = '/gpu:%d'%i
            with tf.device(device):
                sr_inf, loss, grad, interp, high_res = self.super_resolution(sliced4x[i], sliced3x[i], sliced2x[i], sliced1x[i],  with_summary=False, reuse=True, name='gpu_%d'%i)
            print('sr', sr_inf)
            print('loss', loss)
            # print('grad', grad)
            print('interp', interp)
            sr_infs.append(sr_inf)
            losses.append(loss)
            grads.append(grad)
            interps.append(interp)
            high_ress.append(high_res)

        with tf.name_scope('loss'):
            self.losses['train'] = tf.add_n(losses)
        
        with tf.name_scope('infer'):
            sr_inf = tf.concat(sr_infs, axis=0)
            self.add_node(sr_inf, 'inference')
            interp = tf.concat(interps, axis=0)
            high_res = tf.concat(high_ress, axis=0)
        tf.summary.scalar('lr/train', self.lr['train'])
        with tf.name_scope('cpu_summary'):
            
            res_ref = high_res - interp
            res_inf = high_res - sr_inf      
            err_itp = tf.abs(res_ref)            
            err_inf = tf.abs(res_inf)
            l2_err_itp = tf.reduce_sum(tf.square(err_itp), axis=[1, 2, 3])
            l2_err_inf = tf.reduce_sum(tf.square(err_inf), axis=[1, 2, 3])

            l2_inf = tf.reduce_mean(l2_err_inf)
            l2_itp = tf.reduce_mean(l2_err_itp)
            ratio = tf.reduce_mean(l2_err_inf/(l2_err_itp+1e-3))            
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

        self.feed_dict['train'] = ['data3', 'data2', 'data1', 'data0', 'training']
        self.feed_dict['predict'] = ['data3', 'training']
        self.feed_dict['summary'] = ['data3', 'data2', 'data1', 'data0','training']

        self.run_op['train'] = {'train_step': self.train_steps['train'],
                                'loss': self.losses['train'], 'global_step': self.gs}
        self.run_op['predict'] = {'inference': sr_inf, 'interp': interp}
        self.run_op['summary'] = {'summary_all': self.summary_ops['all']}
    
    def _set_train(self):
        pass
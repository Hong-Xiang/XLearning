from collections import defaultdict, OrderedDict, ChainMap
import re
import numpy as np
import tensorflow as tf

from xlearn.utils.collections import DefaultDict


from ..utils.general import with_config
from tensorflow.contrib.framework import arg_scope, add_arg_scope
from ..models.image import conv2d, upsampling2d, super_resolution_infer, repeat, psnr, rmse, stem
from ..models import image as layers
from .base import Net
from xlearn.utils.prints import pp_json
import pprint

class SRNetBase(Net):
    """ Base class for super resolution nets.
    Nodes:
        <image_keys>_<xN>
    """
    @with_config
    def __init__(self,
                 low_shape,
                 nb_down_sample,
                 is_down_sample_0=True,
                 is_down_sample_1=True,
                 upsampling_method='bilinear',
                 image_keys=('data', 'label'),                 
                 crop_size=None,
                 **kwargs):
        Net.__init__(self, **kwargs)
        self.params['is_down_sample_0'] = is_down_sample_0
        self.params['is_down_sample_1'] = is_down_sample_1        
        self.params['nb_down_sample'] = nb_down_sample

        self.params['image_keys'] = image_keys

        self.params['upsampling_method'] = upsampling_method
        self.params['crop_size'] = crop_size

        down_sample_ratio = [1, 1]
        if self.params['is_down_sample_0']:
            down_sample_ratio[0] *= 2
        if self.params['is_down_sample_1']:
            down_sample_ratio[1] *= 2
        self.params['down_sample_ratio'] = down_sample_ratio
        down_sample_ratio_full = [
            d**self.params['nb_down_sample'] for d in down_sample_ratio]
        self.params['down_sample_ratio_full'] = down_sample_ratio_full

        self.params['low_shape'] = [
            self.params['batch_size']] + list(low_shape) + [1]
        self.params['nb_scale'] = self.params['nb_down_sample'] + 1

        self.params['scale_keys'] = ['x%d' %
                                     (2**i) for i in range(self.params['nb_scale'])]
                                     
        self.params.update_short_cut()
        self.sk = self.params['scale_keys']

        self.params['scale'] = {k: 2**i for i, k in enumerate(self.sk)}

        self.shapes = {self.sk[-1]: self.params['low_shape']}
        for i in reversed(range(0, self.params['nb_down_sample'])):
            sk_pre = self.sk[i + 1]
            shape_pre = list(self.shapes[sk_pre])
            shapes_new = list(shape_pre)
            shapes_new[1] *= self.params['down_sample_ratio'][0]
            shapes_new[2] *= self.params['down_sample_ratio'][1]
            self.shapes[self.sk[i]] = shapes_new
        self.params['shapes'] = dict(self.shapes)

        self.params.update_short_cut()
    
    def _default_proxy_name(self, node_name):        
        pattern = re.compile(r'(\w+)_(x\d+)')
        match = pattern.match(node_name)
        idx = self.sk.index(match[2])
        return '{img_key}{idx}'.format(img_key=match[1], idx=idx)

    def _add_input_nodes(self, proxy_name_fn=None):
        from xlearn.utils.general import add_breakpoint
        """ A quick helper method to construct input nodes.        

        Uses parameters:

        `nb_gpus`, `shapes`, `image_keys`
        Inputs:

        proxy_name_fn: a function returns proxy_name with node_name in format <image_key>_<scale_key>

        Returns:

        `out`:    `dict` of `list` of tensors, for each `key` in images_keys:

            output[key] = [tensor_node, tensor_0, tensor_1, ... tensor_(nb_gpus-1)]

        _Note_: `tensor_node` and `tensor_0` may share same tensor if `nb_gpus == 1`.
        """        
        is_multi_gpu = self.p.nb_gpus > 1        
        out = dict()
        for image_key in self.p.image_keys:
            for i, k in enumerate(self.sk, 1):
                node_name = '{image_key}_{scale_key}'.format(image_key=image_key, scale_key=k)
                proxy_name = proxy_name_fn(node_name) if proxy_name_fn else None
                shape = self.shapes[k]
                tensor_node, tensors_gpu = self.add_node(node_name, shape=shape, proxy_name=proxy_name, is_multi_gpu=is_multi_gpu)
                out[node_name] = [tensor_node] + tensors_gpu
        return out
    
    def _kernel(self, inputs, *, device=None, idx=None, reuse=None, name='kernel'):
        """ Super resolution kernel function.
        Inputs:
            inputs: dict of tensors
            reuse:
            name:

        Returns:
            tensors2concat:
            tensors2sum:
            tensors
        Raises:
        """
        pass

    def _apply_kernel_callback(self, outs_cat, outs_sum, outs_update, device, idx, reuse=None):
        if reuse:
            value = self.losses.get('main')
            if value is None:
                self.losses['main'] = []
            self.losses['main'].append(outs_sum['loss/main'])            
    def _fetch_image_nodes(self, nodes, index):
        out = dict()
        for ik in self.p.image_keys:
            for sk in self.sk:
                node_name = '{image_key}_{scale_key}'.format(image_key=ik, scale_key=sk)
                out.update({node_name: nodes[node_name][index]})
        return out
    def _apply_kernel(self, nodes):
        """ Quick method of apply kernel to nodes for multiple gpus.
        Returns:
        out: A dict of merged tensors
        """
        out = dict()
        
        if self.is_multi_gpu:
            with tf.device('/cpu:0'):                                
                _ = self._kernel(self._fetch_image_nodes(nodes, 1), name='kernel')            
            
            outs_cat = DefaultDict(list)
            outs_sum = DefaultDict(list)            
            for i in range(self.p.nb_gpus):                
                with tf.device('/gpu:%d'%i):                    
                    inputs = self._fetch_image_nodes(nodes, i+1)
                    outs_cat_now, outs_sum_now, outs_update = self._kernel(inputs, reuse=True, name='kernel')
                    self._apply_kernel_callback(outs_cat_now, outs_sum_now, outs_update, 'gpu', i, reuse=True)
                    outs_cat.append(outs_cat_now)
                    outs_sum.append(outs_sum_now)        
                    if outs_update is not None:  
                        out.update(outs_update)
        else:            
            outs_cat, outs_sum, outs_update = self._kernel(inputs, name='kernel')
            if outs_update is not None:  
                out.update(outs_update)
        out.update(self._merge(outs_cat, outs_sum))
        return out
        
    def _merge(self, dict_to_concat, dict_to_sum):
        """ Merge tensors from mulitple gpus, by cat or sum.
        """
        with tf.name_scope('merge'):
            merged_cat = dict()            
            with tf.name_scope('concate'):
                for k, v in dict_to_concat.items():            
                    merged_cat[k] = tf.concat(v, axis=0)
        
            merged_sum = dict()            
            with tf.name_scope('summation'):
                for k, v in dict_to_sum.items():
                    merged_sum[k] = tf.add_n(v)
        out = ChainMap(merged_cat, merged_sum)
        return out

    def _set_model(self):
        """ Default implementation of set model of super resolution.            
        """
        nodes = self._add_input_nodes(proxy_name_fn=self._default_proxy_name)
        result = self._apply_kernel(nodes)

        self.add_node('inf', tensor=result['inf'])
        self.add_node('itp', tensor=result['itp'])
        self._add_image_summary(**result)
        
        self.losses['main/sum'] = result['loss/main']
        tf.summary.scalar('loss', self.losses['main/sum'])

        tf.summary.scalar('learning_rate/main', self.lr['main'])
        self.summary_ops['all'] = tf.summary.merge_all()

    def _set_task(self):
        """ Constrcut tasks template for net.
        Tasks like train, evaluate, summary, predict, etc.
        """
        for k in self.train_tasks:
            k_dict = {'train/'+k: 
                {'train': self.train_steps[k],
                 'global_step': self.gs,
                 'loss': self.losses[k+'/sum'] if self.is_multi_gpu else self.losses[k]
                }}
            self.run_op.update(k_dict)

        self.run_op.update({
            'predict': {
                'inf': self.nodes['inf']
            },
            'summary': {
                'summary': self.summary_ops['all']
            },
            'interp': {
                'itp': self.nodes['itp']
            }
        })

        self.feed_dict.update({
            # 'train/main': ['data_'+self.sk[-1], 'label_'+self.sk[0]],
            "train/main": ['data_'+k for k in self.sk] + ['label_'+k for k in self.sk],
            'predict': ['data_'+self.sk[-1]],
            'interp': ['data_'+self.sk[-1]],
            'summary': ['data_'+k for k in self.sk] + ['label_'+k for k in self.sk],
        })

    def _add_image_summary(self, low, high, itp, inf, res, *args, **kwargs):        
        with tf.name_scope('image_summaries'):        
            with tf.name_scope('residual_interpolation'):
                res_ref = high - itp
            
            with tf.name_scope('error'):
                err_inf = tf.abs(high-inf)
                err_ref = tf.abs(res_ref)
            
            with tf.name_scope('metrices'):
                rmse_inf = rmse(high, inf)
                psnr_inf = psnr(high, inf)
                rmse_itp = rmse(high, itp)
                psnr_itp = psnr(high, itp)
                rmse_ratio = rmse_inf/rmse_itp
                psnr_diff = psnr_inf - psnr_itp
    
        tf.summary.image('inference', inf)
        tf.summary.image('residual_inference', res)
        tf.summary.image('interpolation', itp)
        tf.summary.image('residual_reference', res_ref)
        tf.summary.image('error_inference', err_inf)
        tf.summary.image('error_reference', err_ref)
        tf.summary.image('low_resolution', low)
        tf.summary.image('high_resolution', high)

        tf.summary.scalar('rmse/inf', rmse_inf)
        tf.summary.scalar('psnr/inf', psnr_inf)
        tf.summary.scalar('rmse/ref', rmse_itp)
        tf.summary.scalar('psnr/ref', psnr_itp)
        tf.summary.scalar('rmse/ratio', rmse_ratio)
        tf.summary.scalar('psnr/diff', psnr_diff)
        

class SRCNN(SRNetBase):
    """Dong C, Loy CC, He K, Tang X. Image Super-Resolution Using Deep Convolutional Networks. IEEE Trans Pattern Anal Mach Intell. 2016;38(2):295-307. doi:10.1109/TPAMI.2015.2439281."""
    @with_config
    def __init__(self,
                 **kwargs):
        SRNetBase.__init__(self, **kwargs)
        self.params['name'] = "SRCNN"
        self.params.update_short_cut()

    def _kernel(self, inputs, *, device=None, idx=None, reuse=None, name='kernel'):
        """ Super resolution kernel function.
        Inputs:
            inputs: dict of tensors
            reuse:
            name:

        Returns:
            tensors2concat:
            tensors2sum:
            tensors
        Raises:
        """
        with tf.variable_scope(name, 'kernel', reuse=reuse):
            low_res = inputs['data_'+self.sk[-1]]
            high_res = inputs['label_'+self.sk[0]]
            h = low_res
            itp = upsampling2d(
                low_res, size=self.params['down_sample_ratio_full'], method=self.p.upsampling_method)
            with arg_scope([conv2d], activation=tf.nn.elu):
                h = conv2d(itp, 64, kernel_size=9, name='conv0')
                h = conv2d(h, 32, kernel_size=1, name='conv1')
            sri = super_resolution_infer(low_res, h,
                                         size=self.p.down_sample_ratio_full,
                                         method=self.p.upsampling_method,
                                         name='infer')
            inf = sri['inf']
            
            with tf.name_scope('loss/main'):
                loss = tf.losses.mean_squared_error(high_res, inf)

        outs_cat = {
            'low': low_res,
            'high': high_res,
            'itp': itp,
            'inf': inf,
            'res': sri['res']
        }

        outs_sum = {'loss/main': loss}

        return outs_cat, outs_sum, None





class SRVD(SRNetBase):
    """
    [1] Kim J, Lee JK, Lee KM. Accurate Image Super-Resolution Using Very Deep Convolutional Networks. IEEE Trans Pattern Anal Mach Intell. 2015;38(2):295-307. doi:10.1109/TPAMI.2015.2439281.
    """
    @with_config
    def __init__(self,
                 **kwargs):
        SRNetBase.__init__(self, **kwargs)
        self.params['name'] = "SRVD"
        self.params.update_short_cut()

    def _kernel(self, inputs, *, device=None, idx=None, reuse=None, name='kernel'):
        """ Super resolution kernel function.
        Inputs:
            inputs: dict of tensors
            reuse:
            name:

        Returns:
            tensors2concat:
            tensors2sum:
            tensors
        Raises:
        """
        with tf.variable_scope(name, 'kernel', reuse=reuse):
            low_res = inputs['data_'+self.sk[-1]]
            high_res = inputs['label_'+self.sk[0]]
            h = low_res
            itp = upsampling2d(
                low_res, size=self.params['down_sample_ratio_full'], method=self.p.upsampling_method)
            with arg_scope([conv2d], activation=tf.nn.elu):
                h = repeat(itp, conv2d, 20, 'conv2d', filters=64)
            sri = super_resolution_infer(low_res, h,
                                         size=self.p.down_sample_ratio_full,
                                         method=self.p.upsampling_method,
                                         name='infer')
            inf = sri['inf']
            
            with tf.name_scope('loss/main'):
                loss = tf.losses.mean_squared_error(high_res, inf)

        outs_cat = {
            'low': low_res,
            'high': high_res,
            'itp': itp,
            'inf': inf,
            'res': sri['res']
        }

        outs_sum = {'loss/main': loss}

        return outs_cat, outs_sum, None


class SRRes(SRNetBase):
    """
    """
    @with_config
    def __init__(self,
                 filters,
                 depths,
                 scale=(1.0, 1.0),
                 basic_unit=None,
                 nb_units=2,
                 loss_weight=None,
                 **kwargs):
        SRNetBase.__init__(self, **kwargs)
        self.params['filters'] = filters
        self.params['depths'] = depths
        self.params['scale'] = scale
        self.params['name'] = "SRRes"
        self.params['basic_unit'] = basic_unit
        self.params['nb_units'] = nb_units
        self.params['loss_weight'] = loss_weight
        self.params.update_short_cut()
    
    
    def _get_basic_unit_and_args(self, basic_unit_name=None):
        if basic_unit_name is None:
            basic_unit_name = self.p.basic_unit
        if basic_unit_name == 'conv2d':
            unit = layers.conv2d
            args = {'pre_activation': layers.celu}            
        elif basic_unit_name == 'incept':
            unit = layers.incept
            args = {'activation': layers.celu}            
        else:
            raise ValueError('Known ')
        if self.p.is_bn:
            args.update({'normalization': 'bn'})
            args.update({'training': self.training})
        return {'basic_unit': unit, 'basic_unit_args': args, 'scale': self.p.scale}

    def _kernel(self, inputs, reuse=None, name='kernel'):
        with tf.variable_scope(name, 'kernel', reuse=reuse):
            low_res = inputs['data_'+self.sk[-1]]
            high_res = inputs['label_'+self.sk[0]]
            # h = low_res
            # h = stem(low_res, self.p.filters)
            # res_args = self._get_basic_unit_and_args()
            # reuse_infer = None
            # sris = dict()
            # losses = []
            # itp = low_res
            # up_time = 0
            # for i in reversed(range(self.p.nb_down_sample)):
            #     up_time += 1
            #     sk_now = self.sk[i]
            #     with tf.variable_scope('stage_%d'%(i+1)):
            #         h = repeat(h, layers.residual, self.p.depths, res_args, 'res_unit')
            #         itp = upsampling2d(itp, size=self.params['down_sample_ratio'], method=self.p.upsampling_method)
            #         reps = h
            #         for _ in range(up_time):
            #             reps = upsampling2d(reps, size=self.params['down_sample_ratio'], method=self.p.upsampling_method)
            #     sri = super_resolution_infer(low_res, reps,
            #                                 reuse=reuse_infer,
            #                                 size=self.p.down_sample_ratio,
            #                                 method=self.p.upsampling_method,
            #                                 name='infer')
            #     inf0 = sri['inf']
            #     high_res = inputs['label_'+self.sk[i]]
            #     with tf.name_scope('crops'):
            #         crop_size = [0, self.p.crop_size, self.p.crop_size, 0]
            #         with arg_scope([layers.crop], half_crop_size=crop_size):
            #             high_res = layers.crop(high_res, name='high')
            #             for k in sri:
            #                 sri[k] = layers.crop(sri[k], name=k)                        
            #     inf1 = sri['inf']
                
            #     with tf.name_scope('loss/'+sk_now):
            #         loss_now = tf.losses.mean_squared_error(high_res, inf1)
            #     losses.append(loss_now)
            #     low_res = inf0                                    
            #     reuse_infer = True
            # low_res = inputs['data_'+self.sk[-1]]

            # loss_weight = self.p.loss_weight or [1.0]*self.p.nb_down_sample
            # with tf.name_scope('loss/main'):
            #     with tf.name_scope('weight'):
            #         losses_weighted = [l*w for l, w in zip(losses, loss_weight)]
            #     with tf.name_scope('summation'):
            #         loss = tf.add_n(losses_weighted)
            
            # with tf.name_scope('itp_crop'):
            #     itp = layers.crop(itp, half_crop_size=crop_size)


            itp = upsampling2d(
                low_res, size=self.params['down_sample_ratio_full'], method=self.p.upsampling_method)
            h = layers.stem(low_res, self.p.filters)
            res_args = self._get_basic_unit_and_args()
            h = repeat(h, layers.residual, self.p.depths, res_args, 'res_unit')
            h = upsampling2d(h, size=self.params['down_sample_ratio_full'], method=self.p.upsampling_method)
            sri = super_resolution_infer(low_res, h,
                                         size=self.p.down_sample_ratio_full,
                                         method=self.p.upsampling_method,
                                         name='infer')
            inf = sri['inf']
            
            with tf.name_scope('loss/main'):
                loss = tf.losses.mean_squared_error(high_res, inf)

            outs_cat = {
                'low': low_res,
                'high': high_res,
                'itp': itp,
                'inf': sri['inf'],
                'res': sri['res']
            }
            outs_sum = {'loss/main': loss}           
        return outs_cat, outs_sum, None

    def _set_task(self):
        super(SRRes, self)._set_task()
        if self.p.is_bn:
            for k in self.feed_dict:
                self.feed_dict[k] += ['training']

# class SRNet1(SRNetBase):
#     @with_config
#     def __init__(self,
#                  filters=64,
#                  depths=20,
#                  train_verbose=0,
#                  is_bn=True,
#                  is_res=True,
#                  **kwargs):
#         SRNetBase.__init__(self, **kwargs)
#         self.params['name'] = "SRNet1"
#         self.params['filters'] = filters
#         self.params['depths'] = depths
#         self.params['train_verbose'] = train_verbose
#         self.params['is_bn'] = is_bn
#         self.params['is_res'] = is_res
#         self.params.update_short_cut()

#     def super_resolution(self, low_res, high_res, with_summary=False, reuse=None, name=None):
#         with tf.name_scope(name):
#             h = low_res
#             interp = upsampling2d(h, size=self.params['down_sample_ratio'])

#             h = tf.layers.conv2d(interp, 64, 5, padding='same',
#                                  name='conv_stem', activation=tf.nn.elu, reuse=reuse)
#             for i in range(self.params['depths']):
#                 hpre = h
#                 h = tf.layers.conv2d(
#                     h, self.params['filters'], 3, padding='same', name='conv_%d' % i, reuse=reuse)
#                 if self.p.is_bn:
#                     h = tf.layers.batch_normalization(
#                         h, training=self.training, reuse=reuse, name='bn_%d' % i, scale=False)
#                 if self.p.is_res:
#                     h = 0.2 * h + hpre
#                 h = tf.nn.relu(h)

#                 if with_summary:
#                     tf.summary.histogram('activ_%d' % i, h)
#             res_inf = tf.layers.conv2d(
#                 h, 1, 5, padding='same', name='conv_end', reuse=reuse)

#             sr_inf = interp + res_inf

#             res_ref = high_res - interp
#             err_itp = tf.abs(res_ref)
#             err_inf = tf.abs(high_res - sr_inf)
#             # patch_size = self.p.high_shape[1] * self.p.high_shape[2]
#             loss = tf.losses.mean_squared_error(high_res, sr_inf)
#             grad = self.optimizers['train'].compute_gradients(loss)

#             if with_summary:
#                 tf.summary.image('low_res', low_res)
#                 tf.summary.image('high_res', high_res)
#                 tf.summary.image('interp', interp)
#                 tf.summary.image('sr_inf', sr_inf)
#                 tf.summary.image('res_ref', res_ref)
#                 tf.summary.image('res_inf', res_inf)
#                 tf.summary.image('err_itp', err_itp)
#                 tf.summary.image('err_inf', err_inf)
#                 tf.summary.scalar('loss', loss)
#             return sr_inf, loss, grad

#     def _set_model(self):
#         bs_gpu = self.p.batch_size // self.p.nb_gpus
#         sliced_low_res = []
#         sliced_high_res = []
#         with tf.device('/cpu:0'):
#             with tf.name_scope('low_resolution'):
#                 low_res = tf.placeholder(
#                     dtype=tf.float32, shape=self.params['low_shape'], name='low_resolution')
#                 self.add_node(low_res, 'data')
#                 gpu_shape = low_res.shape.as_list()
#                 gpu_shape[0] = bs_gpu
#                 for i in range(self.p.nb_gpus):
#                     with tf.name_scope('device_%d' % i):
#                         sliced_low_res.append(
#                             tf.slice(low_res, [i * bs_gpu, 0, 0, 0], gpu_shape))
#             with tf.name_scope('high_resolution'):
#                 high_res = tf.placeholder(
#                     dtype=tf.float32, shape=self.params['high_shape'], name='high_resolution')
#                 self.add_node(high_res, 'label')
#                 gpu_shape = high_res.shape.as_list()
#                 gpu_shape[0] = bs_gpu
#                 for i in range(self.p.nb_gpus):
#                     with tf.name_scope('device_%d' % i):
#                         sliced_high_res.append(
#                             tf.slice(high_res, [i * bs_gpu, 0, 0, 0], gpu_shape))
#             self.optimizers['train'] = tf.train.AdamOptimizer(self.lr['train'])
#             self.super_resolution(
# sliced_low_res[0], sliced_high_res[0], with_summary=False, reuse=None,
# name='cpu_tower')

#         sr_infs = []
#         losses = []
#         grads = []

#         for i in range(self.p.nb_gpus):
#             device = '/gpu:%d' % i
#             with tf.device(device):
#                 sr_inf, loss, grad = self.super_resolution(
#                     sliced_low_res[i], sliced_high_res[i], with_summary=False, reuse=True, name='gpu_%d' % i)
#             sr_infs.append(sr_inf)
#             losses.append(loss)
#             grads.append(grad)

#         with tf.name_scope('loss'):
#             self.losses['train'] = tf.add_n(losses)

#         with tf.name_scope('infer'):
#             sr_inf = tf.concat(sr_infs, axis=0)
#             self.add_node(sr_inf, 'inference')
#         tf.summary.scalar('lr/train', self.lr['train'])
#         with tf.name_scope('cpu_summary'):
#             interp = upsampling2d(
#                 low_res, size=self.params['down_sample_ratio'])
#             res_ref = high_res - interp
#             res_inf = high_res - sr_inf
#             err_itp = tf.abs(res_ref)
#             err_inf = tf.abs(res_inf)
#             l2_err_itp = tf.reduce_sum(tf.square(err_itp), axis=[1, 2, 3])
#             l2_err_inf = tf.reduce_sum(tf.square(err_inf), axis=[1, 2, 3])

#             l2_inf = tf.reduce_mean(l2_err_inf)
#             l2_itp = tf.reduce_mean(l2_err_itp)
#             ratio = tf.reduce_mean(l2_err_inf / (l2_err_itp + 1e-3))
#             tf.summary.image('low_res', low_res)
#             tf.summary.image('high_res', high_res)
#             tf.summary.image('interp', interp)
#             tf.summary.image('sr_inf', sr_inf)
#             tf.summary.image('res_ref', res_ref)
#             tf.summary.image('res_inf', res_inf)
#             tf.summary.image('err_itp', err_itp)
#             tf.summary.image('err_inf', err_inf)
#             tf.summary.scalar('loss', self.losses['train'])
#             tf.summary.scalar('ratio', ratio)
#             tf.summary.scalar('l2_inf', l2_inf)
#             tf.summary.scalar('l2_itp', l2_itp)

#         train_step = self.train_step(
#             grads, self.optimizers['train'], summary_verbose=self.p.train_verbose)
#         self.train_steps['train'] = train_step
#         self.summary_ops['all'] = tf.summary.merge_all()

#         self.feed_dict['train'] = ['data', 'label', 'training']
#         self.feed_dict['predict'] = ['data', 'training']
#         self.feed_dict['summary'] = ['data', 'label', 'training']

#         self.run_op['train'] = {'train_step': self.train_steps['train'],
#                                 'loss': self.losses['train'], 'global_step': self.gs}
#         self.run_op['predict'] = {'infernce': sr_inf}
#         self.run_op['summary'] = {'summary_all': self.summary_ops['all']}

#     def _set_train(self):
#         pass


# class SRNet2(SRNetBase):
#     @with_config
#     def __init__(self,
#                  filters=64,
#                  depths=2,
#                  hidden_num=8,
#                  train_verbose=0,
#                  **kwargs):
#         SRNetBase.__init__(self, **kwargs)
#         self.params['name'] = "SRNet1"
#         self.params['filters'] = filters
#         self.params['depths'] = depths
#         self.params['train_verbose'] = train_verbose
#         self.params['hidden_num'] = hidden_num
#         self.params.update_short_cut()

#     def gen(self, z, low_res, reuse=None):
#         hidden_num = self.p.hidden_num
#         filters = self.p.filters
#         with tf.variable_scope('gen', reuse=reuse) as vs:
#             with tf.name_scope('z1d2img8x8'):
#                 num_output = int(np.prod([8, 8, self.p.filters]))
#                 h = tf.layers.dense(
#                     z, num_output, activation_fn=None, name='dense')
#                 h = tf.reshape(
#                     x, shape=[-1, 8, 8, filters], name='reshape_latent')
#             with tf.name_scope('img8x8'):
#                 h = residual2(h, filters, name='res_0')
#                 h = residual2(h, filters, name='res_1')
#             h = tf.image.resize_nearest_neighbor(h, size=[16, 8])
#             h = residual2(h, filters, name='res_2')
#             h = residual2(h, filters, name='res_3')
#             h = tf.image.resize_nearest_neighbor(h, size=[32, 8])
#             h = residual2(h, filters, name='res_4')
#             h = residual2(h, filters, name='res_5')
#             h = tf.image.resize_nearest_neighbor(h, size=[64, 8])
#             h = [h, low_res]
#             h = tf.concat(h, axis=-1)
#             h = tf.layers.conv2d(h, filters, 3, padding='same',
#                                  name='conv_6', activation=tf.nn.elu)
#             h = tf.layers.conv2d(h, filters, 3, padding='same',
#                                  name='conv_7', activation=tf.nn.elu)
#             h = tf.image.resize_nearest_neighbor(h, size=[64, 16])
#             h = tf.layers.conv2d(h, filters, 3, padding='same',
#                                  name='conv_8', activation=tf.nn.elu)
#             h = tf.layers.conv2d(h, filters, 3, padding='same',
#                                  name='conv_9', activation=tf.nn.elu)
#             out = tf.layers.conv2d(
#                 h, filters, 1, padding='same', name='conv_10')
#         variables = tf.contrib.framework.get_variables(vs)
#         return out, variables

#     def dis(self, x, low_res, reuse=None):
#         hidden_num = self.p.hidden_num
#         filters = self.p.filters
#         with tf.name_scope('dis'):
#             with tf.name_scope('enc'):
#                 with tf.variable_scope('enc', reuse=reuse) as vs0:
#                     h = tf.layers.conv2d(
#                         h, filters, 3, padding='same', name='conv_0', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters, 3, padding='same', name='conv_1', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters, 3, (1, 2), padding='same', name='conv_2', activation=tf.nn.elu)
#                     h = [h, low_res]
#                     h = tf.concat(h, axis=-1)
#                     h = tf.layers.conv2d(
#                         h, filters, 3, padding='same', name='conv_3', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters, 3, padding='same', name='conv_4', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters * 2, 3, (2, 1), padding='same', name='conv_5', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters * 2, 3, padding='same', name='conv_6', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters * 2, 3, padding='same', name='conv_7', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters * 3, 3, 2, padding='same', name='conv_8', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters * 3, 3, padding='same', name='conv_9', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters * 3, 3, padding='same', name='conv_10', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters * 3, 3, 2, padding='same', name='conv_11', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters * 4, 3, padding='same', name='conv_9', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters * 4, 3, padding='same', name='conv_10', activation=tf.nn.elu)
#                     h = tf.contrib.slim.flatten(h)
#                     embed = tf.layers.dense(h, hidden_num, name='dense')
#                 variables = tf.contrib.framework.get_variables(vs0)
#             with tf.name_scope('dec'):
#                 with tf.variable_scope('dec', reuse=reuse) as vs1:
#                     h = tf.layers.dense(embed, num_output,
#                                         activation_fn=None, name='dense')
#                     h = tf.reshape(
#                         x, shape=[-1, 8, 8, filters], name='reshape_latent')
#                     h = tf.layers.conv2d(
#                         h, filters, 3, padding='same', name='conv_0', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters, 3, padding='same', name='conv_1', activation=tf.nn.elu)
#                     h = tf.image.resize_nearest_neighbor(h, size=[16, 16])
#                     h = tf.layers.conv2d(
#                         h, filters, 3, padding='same', name='conv_2', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters, 3, padding='same', name='conv_3', activation=tf.nn.elu)
#                     h = tf.image.resize_nearest_neighbor(h, size=[32, 32])
#                     h = tf.layers.conv2d(
#                         h, filters, 3, padding='same', name='conv_4', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters, 3, padding='same', name='conv_5', activation=tf.nn.elu)
#                     h = tf.image.resize_nearest_neighbor(h, size=[64, 32])
#                     h = [h, low_res]
#                     h = tf.concat(h, axis=-1)
#                     h = tf.layers.conv2d(
#                         h, filters, 3, padding='same', name='conv_6', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters, 3, padding='same', name='conv_7', activation=tf.nn.elu)
#                     h = tf.image.resize_nearest_neighbor(h, size=[64, 64])
#                     h = tf.layers.conv2d(
#                         h, filters, 3, padding='same', name='conv_8', activation=tf.nn.elu)
#                     h = tf.layers.conv2d(
#                         h, filters, 3, padding='same', name='conv_9', activation=tf.nn.elu)
#                     out = tf.layers.conv2d(
#                         h, filters, 1, padding='same', name='conv_10')
#                 variables.append(tf.contrib.framework.get_variables(vs1))
#         return out, variables

#     def _set_model(self):
#         with tf.name_scope('low_resolution'):
#             low_res = tf.placeholder(
#                 dtype=tf.float32, shape=self.params['low_shape'], name='low_resolution')
#             self.add_node(low_res, 'data')
#         with tf.name_scope('high_resolution'):
#             high_res = tf.placeholder(
#                 dtype=tf.float32, shape=self.params['high_shape'], name='high_resolution')
#             self.add_node(high_res, 'label')
#         self.optimizers['train'] = tf.train.RMSPropOptimizer(self.lr['train'])
#         interp = upsampling2d(
#             low_res, size=self.params['down_sample_ratio'], name='interp_low_res')
#         h = tf.layers.conv2d(low_res, self.p.filters, 5,
#                              padding='same', activation=tf.nn.elu, name='conv_stem')
#         for i in range(self.p.depths):
#             h = tf.layers.conv2d(
#                 h, self.p.filters, 3, padding='same', activation=tf.nn.elu, name='conv_%d' % i)
#         rep_int = upsampling2d(
#             h, size=self.params['down_sample_ratio'], name='interp_reps')
#         res_inf = tf.layers.conv2d(
#             h, self.p.filters, 5, padding='same', activation=tf.nn.elu, name='conv_end' % i)
#         sr_inf = res_inf + interp

#         loss = tf.losses.mean_squared_error(high_res, sr_inf)

#         self.losses['train'] = loss

#         train_step = self.optimizers['train'].minimize(loss)

#         res_ref = high_res - interp
#         res_inf = high_res - sr_inf
#         err_itp = tf.abs(res_ref)
#         err_inf = tf.abs(res_inf)
#         l2_err_itp = tf.reduce_sum(tf.square(err_itp), axis=[1, 2, 3])
#         l2_err_inf = tf.reduce_sum(tf.square(err_inf), axis=[1, 2, 3])

#         l2_inf = tf.reduce_mean(l2_err_inf)
#         l2_itp = tf.reduce_mean(l2_err_itp)
#         ratio = tf.reduce_mean(l2_err_inf / (l2_err_itp + 1e-3))
#         tf.summary.image('low_res', low_res)
#         tf.summary.image('high_res', high_res)
#         tf.summary.image('interp', interp)
#         tf.summary.image('sr_inf', sr_inf)
#         tf.summary.image('res_ref', res_ref)
#         tf.summary.image('res_inf', res_inf)
#         tf.summary.image('err_itp', err_itp)
#         tf.summary.image('err_inf', err_inf)
#         tf.summary.scalar('loss', self.losses['train'])
#         tf.summary.scalar('ratio', ratio)
#         tf.summary.scalar('l2_inf', l2_inf)
#         tf.summary.scalar('l2_itp', l2_itp)

#         self.train_steps['train'] = train_step
#         self.summary_ops['all'] = tf.summary.merge_all()

#         self.feed_dict['train'] = ['data', 'label']
#         self.feed_dict['predict'] = ['data']
#         self.feed_dict['summary'] = ['data', 'label']

#         self.run_op['train'] = {'train_step': self.train_steps['train'],
#                                 'loss': self.losses['train'], 'global_step': self.gs}
#         self.run_op['predict'] = {'infernce': sr_inf}
#         self.run_op['summary'] = {'summary_all': self.summary_ops['all']}

#     def _set_train(self):
#         pass


# class SRNet3(SRNetBase):
#     @with_config
#     def __init__(self,
#                  filters=64,
#                  depths=20,
#                  train_verbose=0,
#                  is_bn=True,
#                  is_res=True,
#                  **kwargs):
#         SRNetBase.__init__(self, **kwargs)
#         self.params['name'] = "SRNet3"
#         self.params['filters'] = filters
#         self.params['depths'] = depths
#         self.params['train_verbose'] = train_verbose
#         self.params['is_bn'] = is_bn
#         self.params['is_res'] = is_res
#         self.params.update_short_cut()

#     def super_resolution(self, low_res, high_res, with_summary=False, reuse=None, name=None):
#         with tf.name_scope(name):
#             h = low_res
#             interp = upsampling2d(h, size=self.params['down_sample_ratio'])

#             h = tf.layers.conv2d(interp,  self.params['filters'], 5, padding='same',
#                                  name='conv_stem', activation=tf.nn.elu, reuse=reuse)
#             for i in range(self.params['depths']):
#                 hpre = h
#                 h = tf.layers.conv2d(
#                     h, self.params['filters'], 3, padding='same', name='conv_%d' % i, reuse=reuse)
#                 if self.p.is_bn:
#                     h = tf.layers.batch_normalization(
#                         h, training=self.training, reuse=reuse, name='bn_%d' % i, scale=False)
#                 if self.p.is_res:
#                     h = 0.2 * h + hpre
#                 h = tf.nn.elu(h)

#                 if with_summary:
#                     tf.summary.histogram('activ_%d' % i, h)
#             res_inf = tf.layers.conv2d(
#                 h, 1, 5, padding='same', name='conv_end', use_bias=True, reuse=reuse)
#             sr_inf = interp + res_inf

#             res_inf, sr_inf, interp = align_by_crop(
#                 high_res, [res_inf, sr_inf, interp])

#             err_inf = tf.abs(high_res - sr_inf)
#             res_ref = high_res - interp
#             err_itp = tf.abs(res_ref)

#             patch_size = self.p.high_shape[1] * self.p.high_shape[2]
#             loss = tf.losses.mean_squared_error(high_res, sr_inf)
#             grad = self.optimizers['train'].compute_gradients(loss)

#             if with_summary:
#                 tf.summary.image('low_res', low_res)
#                 tf.summary.image('high_res', high_res)
#                 tf.summary.image('interp', interp)
#                 tf.summary.image('sr_inf', sr_inf)
#                 tf.summary.image('res_ref', res_ref)
#                 tf.summary.image('res_inf', res_inf)
#                 tf.summary.image('err_itp', err_itp)
#                 tf.summary.image('err_inf', err_inf)
#                 tf.summary.scalar('loss', loss)
#             return sr_inf, loss, grad, interp, high_res

#     def _set_model(self):
#         bs_gpu = self.p.batch_size // self.p.nb_gpus
#         sliced_low_res = []
#         sliced_high_res = []
#         with tf.device('/cpu:0'):
#             with tf.name_scope('low_resolution'):
#                 low_res = tf.placeholder(
#                     dtype=tf.float32, shape=self.params['low_shape'], name='low_resolution')
#                 self.add_node(low_res, 'data')
#                 gpu_shape = low_res.shape.as_list()
#                 gpu_shape[0] = bs_gpu
#                 for i in range(self.p.nb_gpus):
#                     with tf.name_scope('device_%d' % i):
#                         sliced_low_res.append(
#                             tf.slice(low_res, [i * bs_gpu, 0, 0, 0], gpu_shape))
#             with tf.name_scope('high_resolution'):
#                 high_res = tf.placeholder(
#                     dtype=tf.float32, shape=self.params['high_shape'], name='high_resolution')
#                 self.add_node(high_res, 'label')
#                 gpu_shape = high_res.shape.as_list()
#                 gpu_shape[0] = bs_gpu
#                 gpu_shape[1] -= 2 * self.p.crop_size
#                 gpu_shape[2] -= 2 * self.p.crop_size
#                 for i in range(self.p.nb_gpus):
#                     with tf.name_scope('device_%d' % i):
#                         sliced_high_res.append(tf.slice(
#                             high_res, [i * bs_gpu, self.p.crop_size, self.p.crop_size, 0], gpu_shape))
#             self.optimizers['train'] = tf.train.AdamOptimizer(self.lr['train'])
#             self.super_resolution(
# sliced_low_res[0], sliced_high_res[0], with_summary=False, reuse=None,
# name='cpu_tower')

#         sr_infs = []
#         losses = []
#         grads = []
#         interps = []
#         high_ress = []
#         for i in range(self.p.nb_gpus):
#             device = '/gpu:%d' % i
#             with tf.device(device):
#                 sr_inf, loss, grad, interp, high_res = self.super_resolution(
#                     sliced_low_res[i], sliced_high_res[i], with_summary=False, reuse=True, name='gpu_%d' % i)
#             sr_infs.append(sr_inf)
#             losses.append(loss)
#             grads.append(grad)
#             interps.append(interp)
#             high_ress.append(high_res)

#         with tf.name_scope('loss'):
#             self.losses['train'] = tf.add_n(losses)

#         with tf.name_scope('infer'):
#             sr_inf = tf.concat(sr_infs, axis=0)
#             self.add_node(sr_inf, 'inference')
#             interp = tf.concat(interps, axis=0)
#             high_res = tf.concat(high_ress, axis=0)
#         tf.summary.scalar('lr/train', self.lr['train'])
#         with tf.name_scope('cpu_summary'):

#             res_ref = high_res - interp
#             res_inf = high_res - sr_inf
#             err_itp = tf.abs(res_ref)
#             err_inf = tf.abs(res_inf)
#             l2_err_itp = tf.reduce_sum(tf.square(err_itp), axis=[1, 2, 3])
#             l2_err_inf = tf.reduce_sum(tf.square(err_inf), axis=[1, 2, 3])

#             l2_inf = tf.reduce_mean(l2_err_inf)
#             l2_itp = tf.reduce_mean(l2_err_itp)
#             ratio = tf.reduce_mean(l2_err_inf / (l2_err_itp + 1e-3))
#             tf.summary.image('low_res', low_res)
#             tf.summary.image('high_res', high_res)
#             tf.summary.image('interp', interp)
#             tf.summary.image('sr_inf', sr_inf)
#             tf.summary.image('res_ref', res_ref)
#             tf.summary.image('res_inf', res_inf)
#             tf.summary.image('err_itp', err_itp)
#             tf.summary.image('err_inf', err_inf)
#             tf.summary.scalar('loss', self.losses['train'])
#             tf.summary.scalar('ratio', ratio)
#             tf.summary.scalar('l2_inf', l2_inf)
#             tf.summary.scalar('l2_itp', l2_itp)

#         train_step = self.train_step(
#             grads, self.optimizers['train'], summary_verbose=self.p.train_verbose)
#         self.train_steps['train'] = train_step
#         self.summary_ops['all'] = tf.summary.merge_all()

#         self.feed_dict['train'] = ['data', 'label', 'training']
#         self.feed_dict['predict'] = ['data', 'training']
#         self.feed_dict['summary'] = ['data', 'label', 'training']

#         self.run_op['train'] = {'train_step': self.train_steps['train'],
#                                 'loss': self.losses['train'], 'global_step': self.gs}
#         self.run_op['predict'] = {'inference': sr_inf, 'interp': interp}
#         self.run_op['summary'] = {'summary_all': self.summary_ops['all']}

#     def _set_train(self):
#         pass


# class SRNet4(SRNetBase):
#     @with_config
#     def __init__(self,
#                  filters=64,
#                  depths=20,
#                  train_verbose=0,
#                  is_bn=True,
#                  is_res=True,
#                  res_scale=0.1,
#                  **kwargs):
#         SRNetBase.__init__(self, **kwargs)
#         self.params['name'] = "SRNet4"
#         self.params['filters'] = filters
#         self.params['depths'] = depths
#         self.params['train_verbose'] = train_verbose
#         self.params['is_bn'] = is_bn
#         self.params['is_res'] = is_res
#         self.params['res_scale'] = res_scale
#         if self.params['down_sample_ratio'][0] > 1:
#             self.params['down_sample_ratio'][0] = 2
#         if self.params['down_sample_ratio'][1] > 1:
#             self.params['down_sample_ratio'][1] = 2
#         self.params.update_short_cut()

#     def super_resolution(self, img8x, img4x, img2x, img1x, with_summary=False, reuse=None, name=None):
#         cid = 0
#         filters = self.p.filters
#         scale = self.p.res_scale
#         with tf.name_scope(name):

#             with tf.name_scope('net8x4x'):
#                 h = tf.layers.conv2d(img8x,  self.params['filters'], 5, padding='same',
#                                      name='conv_stem', activation=tf.nn.elu, reuse=reuse)
#                 for i in range(self.params['depths'] // 3):
#                     h = residual2(h, filters, name='res_u_%d' %
#                                   cid, reuse=reuse, scale=scale)
#                     cid += 1
#                 h = upsampling2d(h, size=[2, 2])
#                 res4x = tf.layers.conv2d(
#                     h, 1, 5, padding='same', name='conv_4x',  use_bias=True, reuse=reuse)
#                 itp4x = upsampling2d(img8x, size=[2, 2])
#                 inf4x = res4x + itp4x

#             with tf.name_scope('net8x4x'):
#                 for i in range(self.params['depths'] // 3):
#                     h = residual2(h, filters, name='res_u_%d' %
#                                   cid, reuse=reuse, scale=scale)
#                     cid += 1
#                 itp2x = upsampling2d(itp4x, size=[2, 2])
#                 h = upsampling2d(h, size=[2, 2])
#                 res2x = tf.layers.conv2d(
#                     h, 1, 5, padding='same', name='conv_2x', reuse=reuse, use_bias=True,)
#                 inf2x = res2x + itp2x

#             with tf.name_scope('net8x4x'):
#                 for i in range(self.params['depths'] // 3):
#                     h = residual2(h, filters, name='res_u_%d' %
#                                   cid, reuse=reuse, scale=scale)
#                     cid += 1
#                 itp1x = upsampling2d(itp2x, size=[2, 2])
#                 h = upsampling2d(h, size=[2, 2])
#                 res1x = tf.layers.conv2d(
#                     h, 1, 5, padding='same', name='conv_1x', use_bias=True, reuse=reuse)
#                 inf1x = res1x + itp1x

#             with tf.name_scope('crop'):
#                 shape4x = img4x.shape.as_list()
#                 shape2x = img2x.shape.as_list()
#                 shape1x = img1x.shape.as_list()
#                 shape4x[1] -= self.p.crop_size // 2
#                 shape4x[2] -= self.p.crop_size // 2
#                 shape2x[1] -= self.p.crop_size
#                 shape2x[2] -= self.p.crop_size
#                 shape1x[1] -= self.p.crop_size * 2
#                 shape1x[2] -= self.p.crop_size * 2
#                 img4x = tf.slice(
#                     img4x, [0, self.p.crop_size // 4, self.p.crop_size // 4, 0], shape4x)
#                 img2x = tf.slice(
#                     img2x, [0, self.p.crop_size // 2, self.p.crop_size // 2, 0], shape2x)
#                 img1x = tf.slice(
# img1x, [0, self.p.crop_size, self.p.crop_size, 0], shape1x)

#                 inf4x, res4x, itp4x = align_by_crop(
#                     img4x, [inf4x, res4x, itp4x])
#                 inf2x, res2x, itp2x = align_by_crop(
#                     img2x, [inf2x, res2x, itp2x])
#                 inf1x, res1x, itp1x = align_by_crop(
#                     img1x, [inf1x, res1x, itp1x])
#                 # res_inf, sr_inf, interp = align_by_crop(img1x, [res_inf, sr_inf, interp])

#             with tf.name_scope('loss'):
#                 loss4x = tf.losses.mean_squared_error(img4x, inf4x)
#                 loss2x = tf.losses.mean_squared_error(img2x, inf2x)
#                 loss1x = tf.losses.mean_squared_error(img1x, inf1x)
#                 loss = 0.1 * loss4x + 0.5 * loss2x + loss1x
#                 grad = self.optimizers['train'].compute_gradients(loss)

#             return inf1x, loss, grad, itp1x, img1x

#     def add_data(self, nb_down, shape):
#         sliced = []
#         bs_gpu = self.p.batch_size // self.p.nb_gpus
#         name = 'data%d' % nb_down
#         with tf.name_scope(name):
#             data = tf.placeholder(
#                 dtype=tf.float32, shape=shape, name=name)
#             self.add_node(data, name)
#             gpu_shape = data.shape.as_list()
#             gpu_shape[0] = bs_gpu
#             for i in range(self.p.nb_gpus):
#                 with tf.name_scope('device_%d' % i):
#                     sliced.append(
#                         tf.slice(data, [i * bs_gpu, 0, 0, 0], gpu_shape))
#         return sliced

#     def _set_model(self):
#         bs_gpu = self.p.batch_size // self.p.nb_gpus

#         sliced_data0 = []
#         sliced_data1 = []
#         sliced_data2 = []
#         sliced_data3 = []
#         shape4x = self.params['low_shape']
#         shape3x = list(shape4x)
#         shape3x[1] *= 2
#         shape3x[2] *= 2
#         shape2x = list(shape3x)
#         shape2x[1] *= 2
#         shape2x[2] *= 2
#         shape1x = list(shape2x)
#         shape1x[1] *= 2
#         shape1x[2] *= 2

#         with tf.device('/cpu:0'):
#             sliced4x = self.add_data(3, shape4x)
#             sliced3x = self.add_data(2, shape3x)
#             sliced2x = self.add_data(1, shape2x)
#             sliced1x = self.add_data(0, shape1x)
#             if self.p.optimizer_name == 'Adam':
#                 self.optimizers['train'] = tf.train.AdamOptimizer(
#                     self.lr['train'])
#             elif self.p.optimizer_name == 'rmsporp':
#                 self.optimizers['train'] = tf.train.RMSPropOptimizer(
#                     self.lr['train'])
#             self.super_resolution(sliced4x[0], sliced3x[0], sliced2x[0],
# sliced1x[0], with_summary=False, reuse=None, name='cpu_tower')

#         sr_infs = []
#         losses = []
#         grads = []
#         interps = []
#         high_ress = []
#         for i in range(self.p.nb_gpus):
#             device = '/gpu:%d' % i
#             with tf.device(device):
#                 sr_inf, loss, grad, interp, high_res = self.super_resolution(
#                     sliced4x[i], sliced3x[i], sliced2x[i], sliced1x[i],  with_summary=False, reuse=True, name='gpu_%d' % i)
#             print('sr', sr_inf)
#             print('loss', loss)
#             # print('grad', grad)
#             print('interp', interp)
#             sr_infs.append(sr_inf)
#             losses.append(loss)
#             grads.append(grad)
#             interps.append(interp)
#             high_ress.append(high_res)

#         with tf.name_scope('loss'):
#             self.losses['train'] = tf.add_n(losses)

#         with tf.name_scope('infer'):
#             sr_inf = tf.concat(sr_infs, axis=0)
#             self.add_node(sr_inf, 'inference')
#             interp = tf.concat(interps, axis=0)
#             high_res = tf.concat(high_ress, axis=0)
#         tf.summary.scalar('lr/train', self.lr['train'])
#         with tf.name_scope('cpu_summary'):

#             res_ref = high_res - interp
#             res_inf = high_res - sr_inf
#             err_itp = tf.abs(res_ref)
#             err_inf = tf.abs(res_inf)
#             l2_err_itp = tf.reduce_sum(tf.square(err_itp), axis=[1, 2, 3])
#             l2_err_inf = tf.reduce_sum(tf.square(err_inf), axis=[1, 2, 3])

#             l2_inf = tf.reduce_mean(l2_err_inf)
#             l2_itp = tf.reduce_mean(l2_err_itp)
#             ratio = tf.reduce_mean(l2_err_inf / (l2_err_itp + 1e-3))
#             tf.summary.image('high_res', high_res)
#             tf.summary.image('interp', interp)
#             tf.summary.image('sr_inf', sr_inf)
#             tf.summary.image('res_ref', res_ref)
#             tf.summary.image('res_inf', res_inf)
#             tf.summary.image('err_itp', err_itp)
#             tf.summary.image('err_inf', err_inf)
#             tf.summary.scalar('loss', self.losses['train'])
#             tf.summary.scalar('ratio', ratio)
#             tf.summary.scalar('l2_inf', l2_inf)
#             tf.summary.scalar('l2_itp', l2_itp)

#         train_step = self.train_step(
#             grads, self.optimizers['train'], summary_verbose=0)
#         self.train_steps['train'] = train_step
#         self.summary_ops['all'] = tf.summary.merge_all()

#         self.feed_dict['train'] = [
#             'data3', 'data2', 'data1', 'data0', 'training']
#         self.feed_dict['predict'] = ['data3', 'training']
#         self.feed_dict['summary'] = [
#             'data3', 'data2', 'data1', 'data0', 'training']

#         self.run_op['train'] = {'train_step': self.train_steps['train'],
#                                 'loss': self.losses['train'], 'global_step': self.gs}
#         self.run_op['predict'] = {'inference': sr_inf, 'interp': interp}
#         self.run_op['summary'] = {'summary_all': self.summary_ops['all']}

#     def _set_train(self):
#         pass


# class SRNet5(SRNetBase):
#     @with_config
#     def __init__(self,
#                  filters=64,
#                  depths=20,
#                  train_verbose=0,
#                  is_bn=True,
#                  is_res=True,
#                  is_inc=True,
#                  res_scale=0.1,
#                  loss_name='mse',
#                  is_norm=False,
#                  is_norm_gamma=False,
#                  is_poi=False,
#                  is_ada=False,
#                  ada_coe=0.5,
#                  **kwargs):
#         SRNetBase.__init__(self, **kwargs)
#         self.params['name'] = "SRNet5"
#         self.params['is_ada'] = is_ada  # Adaptive? loss partial
#         self.params['filters'] = filters
#         self.params['depths'] = depths
#         self.params['train_verbose'] = train_verbose
#         self.params['is_bn'] = is_bn
#         self.params['is_res'] = is_res
#         self.params['res_scale'] = res_scale
#         self.params['ada_coe'] = ada_coe
#         if self.params['down_sample_ratio'][0] > 1:
#             self.params['down_sample_ratio'][0] = 2
#         if self.params['down_sample_ratio'][1] > 1:
#             self.params['down_sample_ratio'][1] = 2
#         self.params.update_short_cut()
#         self.params['is_poi'] = is_poi

#         self.params['nb_scale'] = self.p.nb_down_sample + 1
#         self.params['scale_keys'] = ['x%d' %
#                                      (2**i) for i in range(self.params['nb_scale'])]
#         self.params.update_short_cut()
#         self.sk = self.params['scale_keys']
#         self.params['true_scale'] = dict()
#         for k in self.sk:
#             self.params['true_scale'][k] = int(k[1:])
#         self.shapes = OrderedDict({self.sk[0]: self.p.high_shape})
#         for i in range(1, self.p.nb_scale):
#             shape_pre = list(self.shapes.values())[i - 1]
#             shapes_new = list(shape_pre)
#             shapes_new[1] /= 2
#             shapes_new[2] /= 2
#             self.shapes[self.sk[i]] = shapes_new
#         self.params['shapes'] = dict(self.shapes)
#         self.params.update_short_cut()

#         if self.p.is_poi:
#             self.loss_fn = lambda t, x: tf.reduce_sum(
#                 tf.nn.log_poisson_loss(t, x), name='loss_poisson')
#         else:
#             self.loss_fn = tf.losses.mean_squared_error

#     @add_arg_scope
#     def _sr_kernel(self, low_res, reuse=None, name=None):
#         """ super resolution kernel
#         Inputs:
#             low_res:
#         Returns:
#         Yields:
#         """
#         nf = self.p.filters
#         scale = self.p.res_scale
#         dr = self.p.down_sample_ratio
#         basic_unit = incept
#         normalization = 'bn' if self.p.is_bn else None
#         basic_unit_args = {'normalization': normalization,
#                            'training': self.training, 'activation': celu}
#         residual_args = {
#             'basic_unit_args': basic_unit_args, 'basic_unit': incept}
#         with tf.variable_scope(name, 'super_resolution_net', reuse=reuse):
#             stemt = stem(low_res, filters=nf, name='stem')
#             repA = stack_residuals(stemt, self.p.depths,
#                                    residual_args, name='stack_A')
#             repB = upsampling2d(repA, size=dr)
#             repC = stack_residuals(repB, self.p.depths,
#                                    residual_args, name='stack_C')
#             inf, res, itp = sr_infer(low_res, repC, dr, name='infer')
#         return inf, res, itp

#     @add_arg_scope
#     def _sr_kernel_dummy(self, low_res, reuse=None, name=None):
#         with tf.variable_scope(name, 'super_resolution_net', reuse=reuse):
#             with tf.name_scope('upsampling'):
#                 itp = upsampling2d(low_res, self.p.down_sample_ratio)
#             with tf.name_scope('infer'):
#                 h = tf.layers.conv2d(itp, 32, 3, padding='same')
#                 inf = tf.layers.conv2d(h, 1, 3, padding='same')
#             with tf.name_scope('res'):
#                 res = inf - itp
#         return inf, res, itp

#     def __crop_tensors(self, infs, itps, ress, ips):
#         infsc = self.__new_tensor_table()
#         itpsc = self.__new_tensor_table()
#         ressc = self.__new_tensor_table()
#         ipsc = OrderedDict()
#         for k in self.sk:
#             ipsc[k] = None
#         with arg_scope([crop], crop_size=self.p.crop_size):
#             for k in self.sk:
#                 ipsc[k] = crop(ips[k], name='crop/ip' + k)
#                 for k_sub in self.sk:
#                     if infs[k][k_sub] is not None:
#                         infsc[k][k_sub] = crop(
#                             infs[k][k_sub], name='crop/inf_' + k + 'to' + k_sub)
#                         itpsc[k][k_sub] = crop(
#                             itps[k][k_sub], name='crop/itp_' + k + 'to' + k_sub)
#                         ressc[k][k_sub] = crop(
#                             ress[k][k_sub], name='crop/res_' + k + 'to' + k_sub)
#         return infsc, itpsc, ressc, ipsc

#     def __new_tensor_table(self):
#         table = OrderedDict()
#         for k in self.sk:
#             table[k] = OrderedDict()
#             for sub_k in self.sk:
#                 table[k][sub_k] = None
#         return table

#     def _super_resolution(self, imgs: dict, reuse=None, name=None):
#         """ Full model of infernce net.
#             One replica on cpu, true position of variables.
#             Multiple replica on gpus.
#         """
#         net_name = 'kernel_net'

#         ips = OrderedDict()
#         for i, k in enumerate(self.sk):
#             ips[k] = imgs[k]
#         with tf.variable_scope(name, 'super_resolution', reuse=reuse):
#             # Construct first kernel net:
#             if not reuse:
#                 with tf.name_scope('dummy_input'):
#                     ipt = tf.placeholder(
#                         dtype=tf.float32, shape=ips[self.sk[1]].shape.as_list())
#                 _ = self._sr_kernel(ipt, name=net_name, reuse=False)

#             # Construct kernel net on inputs of different scale:
#             infs = self.__new_tensor_table()
#             itps = self.__new_tensor_table()
#             ress = self.__new_tensor_table()
#             losses = self.__new_tensor_table()
#             buffer = {k: None for k in self.sk}
#             with arg_scope([self._sr_kernel, self._sr_kernel], name=net_name, reuse=True):
#                 for sk1 in reversed(self.sk):
#                     for sk0 in reversed(self.sk):
#                         if buffer[sk0] is None:
#                             continue
#                         infs[sk0][sk1], ress[sk0][sk1], itps[sk0][sk1] = self._sr_kernel(
#                             buffer[sk0])
#                     for k in infs:
#                         if infs[k][sk1] is not None:
#                             buffer[k] = infs[k][sk1]
#                     buffer[sk1] = ips[sk1]
#             # crop
#             infs, itps, ress, ips = self.__crop_tensors(infs, itps, ress, ips)

#             # Losses
#             with tf.name_scope('losses'):
#                 for sk0 in self.sk:
#                     for sk1 in self.sk:
#                         if infs[sk0][sk1] is not None:
#                             with tf.name_scope('loss_' + sk0 + 'to' + sk1):
#                                 losses[sk0][sk1] = self.loss_fn(
#                                     ips[sk1], infs[sk0][sk1])

#             with tf.name_scope('loss_pre'):
#                 to_add = []
#                 for i in range(self.p.nb_scale - 1):
#                     if losses[self.sk[i + 1]][self.sk[i]] is not None:
#                         if self.p.is_ada:
#                             to_add.append(
#                                 losses[self.sk[i + 1]][self.sk[i]] * (self.p.ada_coe**i))
#                         else:
#                             to_add.append(losses[self.sk[i + 1]][self.sk[i]])
#                 loss_pre = tf.add_n(to_add)
#             with tf.name_scope('loss_all'):
#                 to_add = []
#                 for k0 in self.sk:
#                     for k1 in self.sk:
#                         if losses[k0][k1] is not None:
#                             to_add.append(losses[k0][k1])
#                 loss_all = tf.add_n(to_add)

#             # Gradients
#             with tf.name_scope('grad_pre'):
#                 grad_pre = self.optimizers['train_pre'].compute_gradients(
#                     loss_pre)
#             with tf.name_scope('grad_all'):
#                 grad_all = self.optimizers['train_all'].compute_gradients(
#                     loss_all)

#             # Outs
#             out = {
#                 'loss_pre': loss_pre,
#                 'loss_all': loss_all,
#                 'grad_pre': grad_pre,
#                 'grad_all': grad_all
#             }
#             for k in self.sk:
#                 out.update({'data_' + k: ips[k]})
#                 k0 = self.sk[0]
#                 out.update({'inf_' + k: infs[k][k0]})
#                 out.update({'itp_' + k: itps[k][k0]})
#                 out.update({'res_' + k: ress[k][k0]})
#             out_f = {k: out[k] for k in out if out[k] is not None}
#         return out_f

#     def _add_data(self, nb_down: int, shape: list):
#         """ Add data entry tensor, and split into GPU mini-batch.
#         Inputs:
#             nb_down: tile of down sample, which is used only for naming in graph and nodes.
#             shape: shape of tensor, in cpu (full shape), which should be able to divided by nb_gpus.
#         Returns:
#             A cpu tensor, a list of gpu tensors
#         Yield:
#             None
#         """
#         sliced = []
#         bs_gpu = self.p.batch_size // self.p.nb_gpus
#         name = 'data_%dx' % (2 ** nb_down)
#         name_out = 'data%d' % nb_down
#         with tf.name_scope(name):
#             data = tf.placeholder(
#                 dtype=tf.float32, shape=shape, name=name)
#             self.add_node(data, name_out)
#             gpu_shape = data.shape.as_list()
#             gpu_shape[0] = bs_gpu
#             for i in range(self.p.nb_gpus):
#                 with tf.name_scope('device_%d' % i):
#                     sliced.append(
#                         tf.slice(data, [i * bs_gpu, 0, 0, 0], gpu_shape))
#         return data, sliced

#     def __sum_imgs(self, dict_to_sum):
#         for k, v in dict_to_sum.items():
#             tf.summary.image(k, v, max_outputs=4)

#     def __sum_scas(self, dict_to_sum):
#         for k, v in dict_to_sum.items():
#             tf.summary.scalar(k, v)

#     def __add_analysis(self):
#         with tf.name_scope('analysis'):
#             with tf.name_scope('res_ref'):
#                 res_refs = {k: tf.subtract(
#                     self.datas[self.sk[0]], self.itps[k], name=k) for k in self.itps}
#                 self.__sum_imgs(res_refs)

#             with tf.name_scope('res_inf'):
#                 res_infs = {k: tf.subtract(
#                     self.datas[self.sk[0]], self.infs[k], name=k) for k in self.infs}
#                 self.__sum_imgs(res_infs)

#             with tf.name_scope('err_itp'):
#                 err_itps = {k: tf.abs(res_refs[k], name=k) for k in res_refs}
#                 self.__sum_imgs(err_itps)

#             with tf.name_scope('err_inf'):
#                 err_infs = {k: tf.abs(res_infs[k], name=k) for k in res_infs}
#                 self.__sum_imgs(err_infs)

#             def l2v(d, k): return tf.sqrt(tf.reduce_sum(
#                 tf.square(d[k]), axis=[1, 2, 3]), name=k)
#             with tf.name_scope('l2_itp'):
#                 l2_itps = {k: l2v(err_itps, k) for k in err_itps}
#                 with tf.name_scope('mean'):
#                     l2_itps_mean = {k: tf.reduce_mean(
#                         l2_itps[k]) for k in l2_itps}
#                 self.__sum_scas(l2_itps_mean)
#             with tf.name_scope('l2_inf'):
#                 l2_infs = {k: l2v(err_infs, k) for k in err_infs}
#                 with tf.name_scope('mean'):
#                     l2_infs_mean = {k: tf.reduce_mean(
#                         l2_infs[k]) for k in l2_infs}
#                 self.__sum_scas(l2_infs_mean)
#             with tf.name_scope('ratio'):
#                 def rv(x, y, k): return tf.reduce_mean(
#                     x[k] / (y[k] + 1e-3), name=k)
#                 ratios = {k: rv(l2_infs, l2_itps, k) for k in l2_infs}
#                 self.__sum_scas(ratios)
#         tf.summary.scalar('lr/train_pre', self.lr['train_pre'])
#         tf.summary.scalar('lr/train_all', self.lr['train_all'])

#     def __add_merge(self):
#         outs = self.outs

#         with tf.name_scope('merge'):
#             with tf.name_scope('infers'):
#                 infs = {k[-2:]: tf.concat(outs[k], axis=0, name=k)
#                         for k in outs if 'inf' in k}
#                 for k in infs:
#                     self.add_node(infs[k], 'inf_' + k)
#                 self.infs = infs
#                 self.__sum_imgs(infs)
#             with tf.name_scope('interps'):
#                 itps = {k[-2:]: tf.concat(outs[k], axis=0, name=k)
#                         for k in outs if 'itp' in k}
#                 for k in itps:
#                     self.add_node(itps[k], 'itp_' + k)
#                 self.itps = itps
#                 self.__sum_imgs(itps)
#             with tf.name_scope('labels'):
#                 datas = {k[-2:]: tf.concat(outs[k], axis=0, name=k)
#                          for k in outs if 'data' in k}
#                 self.datas = datas
#                 self.__sum_imgs(datas)

#             with tf.name_scope('loss'):
#                 self.losses['train_pre'] = tf.add_n(
#                     outs['loss_pre'], name='loss_pre')
#                 self.losses['train_all'] = tf.add_n(
#                     outs['loss_all'], name='loss_all')
#             tf.summary.scalar('loss_pre', self.losses['train_pre'])
#             tf.summary.scalar('loss_all', self.losses['train_all'])

#             train_step_pre = self.train_step(
#                 outs['grad_pre'], self.optimizers['train_pre'], summary_verbose=0, name='train_pre')
#             train_step_all = self.train_step(
# outs['grad_all'], self.optimizers['train_all'], summary_verbose=0,
# name='train_all')

#         self.train_steps['train_pre'] = train_step_pre
#         self.train_steps['train_all'] = train_step_all
#         return None

#     def __add_api(self):

#         # self.summary_ops['all'] = tf.summary.merge_all()

#         common_list = ['data' + str(s)
#                        for s in range(self.p.nb_scale)] + ['training']
#         self.feed_dict['default'] = list()
#         self.feed_dict['train_pre'] = list(common_list) + ['lr/train_pre']
#         self.feed_dict['train_all'] = list(common_list) + ['lr/train_all']
#         for i, k in enumerate(self.sk[:-1]):
#             k0 = self.p.true_scale[self.sk[i + 1]]
#             self.feed_dict['predict_%d' % k0] = [
#                 'data%d' % (i + 1), 'training']
#         self.feed_dict['summary'] = list(
#             common_list) + ['lr/train_pre', 'lr/train_all']

#         self.run_op['train_pre'] = {'train_step_pre': self.train_steps['train_pre'],
#                                     'loss': self.losses['train_pre'], 'global_step': self.gs}
#         self.run_op['train_all'] = {'train_step_all': self.train_steps['train_all'],
#                                     'loss': self.losses['train_all'], 'global_step': self.gs}
#         for s in self.sk[1:]:
#             self.run_op['predict_' + s] = {
#                 'inference': self.nodes['inf_' + s], 'interpolation': self.nodes['itp_' + s]}
#         self.run_op['summary'] = {'summary_all': self.summary_ops['all']}

#     def _set_model(self):
#         bs_gpu = self.p.batch_size // self.p.nb_gpus
#         imgs_cpu = dict()
#         imgs_gpus = []
#         for i in range(self.p.nb_gpus):
#             imgs_gpus.append(dict())
#         with tf.device('/cpu:0'):
#             for i, k in enumerate(self.p.scale_keys):
#                 data_cpu, data_gpus = self._add_data(i, self.shapes[k])
#                 imgs_cpu[k] = data_cpu
#                 for j in range(self.p.nb_gpus):
#                     imgs_gpus[j][k] = data_gpus[j]
#             if self.p.optimizer_name == 'Adam':
#                 self.optimizers['train_pre'] = tf.train.AdamOptimizer(
#                     self.lr['train_pre'])
#                 self.optimizers['train_all'] = tf.train.AdamOptimizer(
#                     self.lr['train_all'])
#             elif self.p.optimizer_name == 'rmsporp':
#                 self.optimizers['train_pre'] = tf.train.RMSPropOptimizer(
#                     self.lr['train_pre'])
#                 self.optimizers['train_all'] = tf.train.RMSPropOptimizer(
#                     self.lr['train_all'])
#             self._super_resolution(imgs_gpus[0], name='net_main', reuse=False)

#         # GPU models
#         outs = dict()
#         for i in range(self.p.nb_gpus):
#             device = '/gpu:%d' % i
#             with tf.device(device):
#                 gpu_out = self._super_resolution(
#                     imgs_gpus[i], name='net_main', reuse=True)
#             for k in gpu_out:
#                 item = outs.get(k)
#                 if item is None:
#                     outs[k] = [gpu_out[k]]
#                 else:
#                     outs[k].append(gpu_out[k])

#         self.outs = outs
#         # for k in outs:
#         #     print(k, ': ', outs[k])

#         self.__add_merge()
#         self.__add_analysis()
#         self.summary_ops['all'] = tf.summary.merge_all()
#         self.__add_api()

#     def _set_train(self):
#         pass


# class SRNet6(SRNetBase):
    # @with_config
    # def __init__(self,
    #              filters=64,
    #              depths=20,
    #              is_bn=True,
    #              res_scale=0.1,
    #              loss_name='mse',
    #              basic_unit='incept',
    #              **kwargs):
    #     SRNetBase.__init__(self, **kwargs)
    #     self.params['name'] = "SRNet6"
    #     self.params['filters'] = filters
    #     self.params['depths'] = depths
    #     self.params['is_bn'] = is_bn
    #     self.params['res_scale'] = res_scale
    #     if self.params['down_sample_ratio'][0] > 1:
    #         self.params['down_sample_ratio'][0] = 2
    #     if self.params['down_sample_ratio'][1] > 1:
    #         self.params['down_sample_ratio'][1] = 2
    #     self.params.update_short_cut()
    #     self.params['basic_unit'] = basic_unit

    #     self.params['nb_scale'] = self.p.nb_down_sample + 1
    #     self.params['scale_keys'] = ['x%d' %
    #                                  (2**i) for i in range(self.params['nb_scale'])]
    #     self.params.update_short_cut()
    #     self.sk = self.params['scale_keys']
    #     self.params['true_scale'] = dict()
    #     for k in self.sk:
    #         self.params['true_scale'][k] = int(k[1:])
    #     self.shapes = OrderedDict({self.sk[0]: self.p.high_shape})
    #     for i in range(1, self.p.nb_scale):
    #         shape_pre = list(self.shapes.values())[i - 1]
    #         shapes_new = list(shape_pre)
    #         shapes_new[1] /= 2
    #         shapes_new[2] /= 2
    #         self.shapes[self.sk[i]] = shapes_new
    #     self.params['shapes'] = dict(self.shapes)
    #     self.params.update_short_cut()

    #     if self.p.is_poi:
    #         self.loss_fn = lambda t, x: tf.reduce_sum(
    #             tf.nn.log_poisson_loss(t, x), name='loss_poisson')
    #     else:
    #         self.loss_fn = tf.losses.mean_squared_error

    # @add_arg_scope
    # def _sr_kernel(self, datas_noise, datas_clean, reuse=None, name=None):
    #     """ super resolution kernel
    #     Inputs:
    #         low_res:
    #     Returns:
    #     Yields:
    #     """
    #     nf = self.p.filters
    #     scale = self.p.res_scale
    #     dr = self.p.down_sample_ratio
    #     if self.p.basic_unit == 'incept':
    #         basic_unit = incept
    #     else:
    #         basic_unit = residual
    #     normalization = 'bn' if self.p.is_bn else None
    #     basic_unit_args = {'normalization': normalization,
    #                        'training': self.training, 'activation': celu}
    #     residual_args = {'basic_unit_args': basic_unit_args,
    #                      'basic_unit': basic_unit}
    #     with tf.variable_scope(name, 'super_resolution_net', reuse=reuse):
    #         if is_stem:
    #             stemt = stem(low_res, filters=nf, name='stem')
    #         else:
    #             stemt = low_res
    #         repA = stack_residuals(stemt, self.p.depths,
    #                                residual_args, name='stack_A')
    #         repB = upsampling2d(repA, size=dr)
    #         repC = stack_residuals(repB, self.p.depths,
    #                                residual_args, name='stack_C')
    #         inf, res, itp = sr_infer(low_res, repC, dr, name='infer')
    #     return inf, res, itp, repC

    # @add_arg_scope
    # def _sr_kernel_dummy(self, low_res, reuse=None, name=None):
    #     with tf.variable_scope(name, 'super_resolution_net', reuse=reuse):
    #         with tf.name_scope('upsampling'):
    #             itp = upsampling2d(low_res, self.p.down_sample_ratio)
    #         with tf.name_scope('infer'):
    #             h = tf.layers.conv2d(itp, 32, 3, padding='same')
    #             inf = tf.layers.conv2d(h, 1, 3, padding='same')
    #         with tf.name_scope('res'):
    #             res = inf - itp
    #     return inf, res, itp

    # def __crop_tensors(self, infs, itps, ress, ips):
    #     with arg_scope([crop], crop_size=self.p.crop_size):
    #         ipsc = crop(ips, name='crop/ip')
    #         infc = crop(inf, name='crop/inf')
    #         itpc = crop(itp, name='crop/itp')
    #         resc = crop(res, name='crop/res')
    #     return infc, itpc, resc, ipc

    # def _super_resolution(self, img_n: dict, img_c: dict, reuse=None, name=None):
    #     """ Full model of infernce net.
    #         One replica on cpu, true position of variables.
    #         Multiple replica on gpus.
    #     """
    #     net_name = 'kernel_net'

    #     datas_noise = OrderedDict()
    #     datas_clean = OrderedDict()
    #     for k in self.sk:
    #         datas_noise[k] = img_n[k]
    #         datas_clean[k] = img_c[k]

    #     with tf.variable_scope(name, 'super_resolution', reuse=reuse):
    #         # Construct first kernel net:
    #         if not reuse:
    #             _ = self._sr_kernel(ipt, name=net_name, reuse=False)

    #         # Construct kernel net on inputs of different scale:
    #         losses = self.__new_tensor_table()
    #         buffer = {k: None for k in self.sk}
    #         with arg_scope([self._sr_kernel, self._sr_kernel], name=net_name, reuse=True):
    #             for sk1 in reversed(self.sk):
    #                 for sk0 in reversed(self.sk):
    #                     if buffer[sk0] is None:
    #                         continue
    #                     infs[sk0][sk1], ress[sk0][sk1], itps[sk0][sk1] = self._sr_kernel(
    #                         buffer[sk0])
    #                 for k in infs:
    #                     if infs[k][sk1] is not None:
    #                         buffer[k] = infs[k][sk1]
    #                 buffer[sk1] = ips[sk1]
    #         # crop
    #         infs, itps, ress, ips = self.__crop_tensors(infs, itps, ress, ips)

    #         # Losses
    #         with tf.name_scope('losses'):
    #             for sk0 in self.sk:
    #                 for sk1 in self.sk:
    #                     if infs[sk0][sk1] is not None:
    #                         with tf.name_scope('loss_' + sk0 + 'to' + sk1):
    #                             losses[sk0][sk1] = self.loss_fn(
    #                                 ips[sk1], infs[sk0][sk1])

    #         with tf.name_scope('loss_pre'):
    #             to_add = []
    #             for i in range(self.p.nb_scale - 1):
    #                 if losses[self.sk[i + 1]][self.sk[i]] is not None:
    #                     if self.p.is_ada:
    #                         to_add.append(
    #                             losses[self.sk[i + 1]][self.sk[i]] * (0.5**i))
    #                     else:
    #                         to_add.append(losses[self.sk[i + 1]][self.sk[i]])
    #             loss_pre = tf.add_n(to_add)
    #         with tf.name_scope('loss_all'):
    #             to_add = []
    #             for k0 in self.sk:
    #                 for k1 in self.sk:
    #                     if losses[k0][k1] is not None:
    #                         to_add.append(losses[k0][k1])
    #             loss_all = tf.add_n(to_add)

    #         # Gradients
    #         with tf.name_scope('grad_pre'):
    #             grad_pre = self.optimizers['train_pre'].compute_gradients(
    #                 loss_pre)
    #         with tf.name_scope('grad_all'):
    #             grad_all = self.optimizers['train_all'].compute_gradients(
    #                 loss_all)

    #         # Outs
    #         out = {
    #             'loss_pre': loss_pre,
    #             'loss_all': loss_all,
    #             'grad_pre': grad_pre,
    #             'grad_all': grad_all
    #         }
    #         for k in self.sk:
    #             out.update({'data_' + k: ips[k]})
    #             k0 = self.sk[0]
    #             out.update({'inf_' + k: infs[k][k0]})
    #             out.update({'itp_' + k: itps[k][k0]})
    #             out.update({'res_' + k: ress[k][k0]})
    #         out_f = {k: out[k] for k in out if out[k] is not None}
    #     return out_f

    # def _add_data(self, nb_down: int, shape: list, prefix: str=None):
    #     """ Add data entry tensor, and split into GPU mini-batch.
    #     Inputs:
    #         nb_down: tile of down sample, which is used only for naming in graph and nodes.
    #         shape: shape of tensor, in cpu (full shape), which should be able to divided by nb_gpus.
    #     Returns:
    #         A cpu tensor, a list of gpu tensors
    #     Yield:
    #         None
    #     """
    #     if prefix is None:
    #         prefix = 'data'
    #     sliced = []
    #     bs_gpu = self.p.batch_size // self.p.nb_gpus
    #     name = 'data_%dx' % (2 ** nb_down)
    #     name_out = 'data%d' % nb_down
    #     with tf.name_scope(name):
    #         data = tf.placeholder(
    #             dtype=tf.float32, shape=shape, name=name)
    #         self.add_node(data, name_out)
    #         gpu_shape = data.shape.as_list()
    #         gpu_shape[0] = bs_gpu
    #         for i in range(self.p.nb_gpus):
    #             with tf.name_scope('device_%d' % i):
    #                 sliced.append(
    #                     tf.slice(data, [i * bs_gpu, 0, 0, 0], gpu_shape))
    #     return data, sliced

    # def __sum_imgs(self, dict_to_sum):
    #     for k, v in dict_to_sum.items():
    #         tf.summary.image(k, v, max_outputs=4)

    # def __sum_scas(self, dict_to_sum):
    #     for k, v in dict_to_sum.items():
    #         tf.summary.scalar(k, v)

    # def __add_analysis(self):
    #     with tf.name_scope('analysis'):
    #         with tf.name_scope('res_ref'):
    #             res_refs = {k: tf.subtract(
    #                 self.datas[self.sk[0]], self.itps[k], name=k) for k in self.itps}
    #             self.__sum_imgs(res_refs)

    #         with tf.name_scope('res_inf'):
    #             res_infs = {k: tf.subtract(
    #                 self.datas[self.sk[0]], self.infs[k], name=k) for k in self.infs}
    #             self.__sum_imgs(res_infs)

    #         with tf.name_scope('err_itp'):
    #             err_itps = {k: tf.abs(res_refs[k], name=k) for k in res_refs}
    #             self.__sum_imgs(err_itps)

    #         with tf.name_scope('err_inf'):
    #             err_infs = {k: tf.abs(res_infs[k], name=k) for k in res_infs}
    #             self.__sum_imgs(err_infs)

    #         def l2v(d, k): return tf.sqrt(tf.reduce_sum(
    #             tf.square(d[k]), axis=[1, 2, 3]), name=k)
    #         with tf.name_scope('l2_itp'):
    #             l2_itps = {k: l2v(err_itps, k) for k in err_itps}
    #             with tf.name_scope('mean'):
    #                 l2_itps_mean = {k: tf.reduce_mean(
    #                     l2_itps[k]) for k in l2_itps}
    #             self.__sum_scas(l2_itps_mean)
    #         with tf.name_scope('l2_inf'):
    #             l2_infs = {k: l2v(err_infs, k) for k in err_infs}
    #             with tf.name_scope('mean'):
    #                 l2_infs_mean = {k: tf.reduce_mean(
    #                     l2_infs[k]) for k in l2_infs}
    #             self.__sum_scas(l2_infs_mean)
    #         with tf.name_scope('ratio'):
    #             def rv(x, y, k): return tf.reduce_mean(
    #                 x[k] / (y[k] + 1e-3), name=k)
    #             ratios = {k: rv(l2_infs, l2_itps, k) for k in l2_infs}
    #             self.__sum_scas(ratios)
    #     tf.summary.scalar('lr/train_pre', self.lr['train_pre'])
    #     tf.summary.scalar('lr/train_all', self.lr['train_all'])

    # def __add_merge(self):
    #     outs = self.outs

    #     with tf.name_scope('merge'):
    #         with tf.name_scope('infers'):
    #             infs = {k[-2:]: tf.concat(outs[k], axis=0, name=k)
    #                     for k in outs if 'inf' in k}
    #             for k in infs:
    #                 self.add_node(infs[k], 'inf_' + k)
    #             self.infs = infs
    #             self.__sum_imgs(infs)
    #         with tf.name_scope('interps'):
    #             itps = {k[-2:]: tf.concat(outs[k], axis=0, name=k)
    #                     for k in outs if 'itp' in k}
    #             for k in itps:
    #                 self.add_node(itps[k], 'itp_' + k)
    #             self.itps = itps
    #             self.__sum_imgs(itps)
    #         with tf.name_scope('labels'):
    #             datas = {k[-2:]: tf.concat(outs[k], axis=0, name=k)
    #                      for k in outs if 'data' in k}
    #             self.datas = datas
    #             self.__sum_imgs(datas)

    #         with tf.name_scope('loss'):
    #             self.losses['train_pre'] = tf.add_n(
    #                 outs['loss_pre'], name='loss_pre')
    #             self.losses['train_all'] = tf.add_n(
    #                 outs['loss_all'], name='loss_all')
    #         tf.summary.scalar('loss_pre', self.losses['train_pre'])
    #         tf.summary.scalar('loss_all', self.losses['train_all'])

    #         train_step_pre = self.train_step(
    #             outs['grad_pre'], self.optimizers['train_pre'], summary_verbose=0, name='train_pre')
    #         train_step_all = self.train_step(
    # outs['grad_all'], self.optimizers['train_all'], summary_verbose=0,
    # name='train_all')

    #     self.train_steps['train_pre'] = train_step_pre
    #     self.train_steps['train_all'] = train_step_all
    #     return None

    # def __add_api(self):

    #     # self.summary_ops['all'] = tf.summary.merge_all()

    #     common_list = ['data' + str(s)
    #                    for s in range(self.p.nb_scale)] + ['training']
    #     self.feed_dict['default'] = list()
    #     self.feed_dict['train_pre'] = list(common_list) + ['lr/train_pre']
    #     self.feed_dict['train_all'] = list(common_list) + ['lr/train_all']
    #     for i, k in enumerate(self.sk[:-1]):
    #         k0 = self.p.true_scale[self.sk[i + 1]]
    #         self.feed_dict['predict_%d' % k0] = [
    #             'data%d' % (i + 1), 'training']
    #     self.feed_dict['summary'] = list(
    #         common_list) + ['lr/train_pre', 'lr/train_all']

    #     self.run_op['train_pre'] = {'train_step_pre': self.train_steps['train_pre'],
    #                                 'loss': self.losses['train_pre'], 'global_step': self.gs}
    #     self.run_op['train_all'] = {'train_step_all': self.train_steps['train_all'],
    #                                 'loss': self.losses['train_all'], 'global_step': self.gs}
    #     for s in self.sk[1:]:
    #         self.run_op['predict_' + s] = {
    #             'inference': self.nodes['inf_' + s], 'interpolation': self.nodes['itp_' + s]}
    #     self.run_op['summary'] = {'summary_all': self.summary_ops['all']}

    # def _set_model(self):
    #     bs_gpu = self.p.batch_size // self.p.nb_gpus
    #     imgs_cpu = dict()
    #     imgs_gpus = []
    #     for i in range(self.p.nb_gpus):
    #         imgs_gpus.append(dict())
    #     with tf.device('/cpu:0'):
    #         for i, k in enumerate(self.p.scale_keys):
    #             data_cpu, data_gpus = self._add_data(i, self.shapes[k])
    #             imgs_cpu[k] = data_cpu
    #             for j in range(self.p.nb_gpus):
    #                 imgs_gpus[j][k] = data_gpus[j]
    #         if self.p.optimizer_name == 'Adam':
    #             self.optimizers['train_pre'] = tf.train.AdamOptimizer(
    #                 self.lr['train_pre'])
    #             self.optimizers['train_all'] = tf.train.AdamOptimizer(
    #                 self.lr['train_all'])
    #         elif self.p.optimizer_name == 'rmsporp':
    #             self.optimizers['train_pre'] = tf.train.RMSPropOptimizer(
    #                 self.lr['train_pre'])
    #             self.optimizers['train_all'] = tf.train.RMSPropOptimizer(
    #                 self.lr['train_all'])
    #         self._super_resolution(imgs_gpus[0], name='net_main', reuse=False)

    #     # GPU models
    #     outs = dict()
    #     for i in range(self.p.nb_gpus):
    #         device = '/gpu:%d' % i
    #         with tf.device(device):
    #             gpu_out = self._super_resolution(
    #                 imgs_gpus[i], name='net_main', reuse=True)
    #         for k in gpu_out:
    #             item = outs.get(k)
    #             if item is None:
    #                 outs[k] = [gpu_out[k]]
    #             else:
    #                 outs[k].append(gpu_out[k])

    #     self.outs = outs
    #     # for k in outs:
    #     #     print(k, ': ', outs[k])

    #     self.__add_merge()
    #     self.__add_analysis()
    #     self.summary_ops['all'] = tf.summary.merge_all()
    #     self.__add_api()

    # def _set_train(self):
    #     pass

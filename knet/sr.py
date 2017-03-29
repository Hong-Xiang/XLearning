import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, ELU, LeakyReLU, Conv2D, UpSampling2D, BatchNormalization, Cropping2D, add, Lambda

from keras import backend as K
from keras.engine.topology import Layer


from xlearn.knet.base import KNet

import xlearn.utils.xpipes as utp
from ..utils.general import with_config, enter_debug
from ..utils.tensor import upsample_shape, downsample_shape
from ..keras_ext.models import sub
from ..kmodel.image import conv_blocks


IMAGE_SUMMARY_MAX_OUTPUT = 5
MAX_DOWNSAMPLE = 3


class KNetSR(KNet):
    @with_config
    def __init__(self,
                 is_down_sample_0=True,
                 is_down_sample_1=True,
                 nb_down_sample=MAX_DOWNSAMPLE,
                 is_deconv=False,
                 settings=None,
                 **kwargs):
        super(KNetSR, self).__init__(**kwargs)
        self._settings = settings
        self._is_down_sample_0 = self._update_settings(
            'is_down_sample_0', is_down_sample_0)
        self._is_down_sample_1 = self._update_settings(
            'is_down_sample_1', is_down_sample_1)
        self._nb_down_sample = self._update_settings(
            'nb_down_sample', nb_down_sample)

        self._is_deconv = self._update_settings('is_deconv', is_deconv)

        # # Check settings
        # shape_o_cal = (upsample_shape(
        #     self._inputs_shapes[0][:2], self._down_sample_ratio))
        # shape_o_cal = tuple(shape_o_cal)
        # shape_o_ip = tuple(self._outputs_shapes[0][:2])
        # if shape_o_cal != shape_o_ip:
        #     raise ValueError('Inconsistant shape_i: {0}, shape_o: {1}, and ratio: {2}.'.format(
        #         self._inputs_shapes[0], shape_o_ip, self._down_sample_ratio))

        self._residuals = dict()
        self._res_inf = None
        self._res_ref = None
        self._models_names = ['sr', 'res_itp', 'res_out']
        self._is_trainable = [True, False, False]

        self._down_sample_ratio = [1, 1]
        if self._is_down_sample_0:
            self._down_sample_ratio[0] = 2
        if self._is_down_sample_1:
            self._down_sample_ratio[1] = 2
        self._down_sample_ratio = tuple(self._down_sample_ratio)
        self._update_settings('down_sample_ratio', self._down_sample_ratio)
        self._shapes = []
        self._shapes.append(self._inputs_shapes[0])
        self._down_sample_ratios = []
        self._down_sample_ratios.append([1, 1])

        for i in range(self._nb_down_sample):
            self._shapes.append(downsample_shape(self._shapes[i],
                                                 list(self._down_sample_ratio) + [1]))
            self._down_sample_ratios.append(upsample_shape(self._down_sample_ratios[i],
                                                           self._down_sample_ratio))
        self._update_settings('shapes', self._shapes)
        self._update_settings('down_sample_ratios', self._down_sample_ratios)

    def _define_models(self):
        self._ips = []
        for i in range(self._nb_down_sample + 1):
            with tf.name_scope('input_%dx' % (2**i)) as scope:
                self._ips.append(Input(self._shapes[i]))
        self._ipn = self._ips[self._nb_down_sample]

        self._ups = []
        for i in range(self._nb_down_sample):
            with tf.name_scope('upsample_%dx' % (2**(i + 1))):
                self._ups.append(UpSampling2D(
                    size=self._down_sample_ratio)(self._ips[i + 1]))
        # residual references
        self._rrs = []
        for i in range(self._nb_down_sample):
            with tf.name_scope('residual_reference_%dx' % (2**(i + 1))):
                self._rrs.append(self._ips[i] - self._ups[i])

        with tf.name_scope('interpolation'):
            upl = UpSampling2D(
                size=self._down_sample_ratios[self._nb_down_sample])
            self._upfull = upl(self._ipn)

        with tf.name_scope('res_itp'):
            res_itp = sub(self._ips[0], self._upfull)
        self._models[self.model_id('res_itp')] = Model(self._ips, res_itp)

    @property
    def nb_down_sample(self):
        return self._nb_down_sample


class SRInterp(KNetSR):
    @with_config
    def __init__(self, **kwargs):
        super(SRInterp, self).__init__(**kwargs)

    def _define_models(self):
        super(SRInterp, self)._define_models()
        with tf.name_scope('inference'):
            upl = UpSampling2D(
                size=self._down_sample_ratios[self._nb_down_sample])
            self._output = upl(self._ipn)
        with tf.name_scope('res_out'):
            res_out = sub(self._ips[0], self._output)
        self._is_trainable = [False, False, False]
        self._models[self.model_id('sr')] = Model(self._ips, self._output)
        self._models[self.model_id('res_out')] = Model(self._ips, res_out)


class SRDv0(KNetSR):
    """ based on arxiv 1501.00092 """
    @with_config
    def __init__(self, **kwargs):
        KNetSR.__init__(self, **kwargs)

    def _define_models(self):
        KNetSR._define_models(self)
        with tf.name_scope('upsampling'):
            ups = UpSampling2D(
                size=self._down_sample_ratios[self._nb_down_sample])(self._ipn)
        with tf.name_scope('conv_0'):
            x = Conv2D(64, 9, activation='elu', padding='same')(ups)
        with tf.name_scope('conv_1'):
            x = Conv2D(32, 1, activation='elu', padding='same')(x)
        with tf.name_scope('output'):
            res_inf = Conv2D(1, 5, padding='same')(x)
            img_inf = add([res_inf, ups])
        with tf.name_scope('res_out'):
            res_out = sub(self._ips[0], img_inf)
        self._models[self.model_id('sr')] = Model(self._ipn, img_inf)
        self._models[self.model_id('res_out')] = Model(
            [self._ips[0], self._ipn], res_out)


class SRDv0b(KNetSR):
    @with_config
    def __init__(self, **kwargs):
        KNetSR.__init__(self, **kwargs)

    def _define_models(self):
        KNetSR._define_models(self)
        with tf.name_scope('upsampling'):
            ups = UpSampling2D(
                size=self._down_sample_ratios[self._nb_down_sample])(self._ipn)
        with tf.name_scope('conv_0'):
            x = Conv2D(64, 9, activation='elu', padding='same')(ups)
            x = BatchNormalization()(x)
        with tf.name_scope('conv_1'):
            x = Conv2D(32, 1, activation='elu', padding='same')(x)
            x = BatchNormalization()(x)
        with tf.name_scope('output'):
            res_inf = Conv2D(1, 5, padding='same')(x)
            img_inf = add([res_inf, ups])
        with tf.name_scope('res_out'):
            res_out = sub(self._ips[0], img_inf)
        self._models[self.model_id('sr')] = Model(self._ipn, img_inf)
        self._models[self.model_id('res_out')] = Model(
            [self._ips[0], self._ipn], res_out)


class SRDv1(KNetSR):
    """ based on arxiv Accurate Image Super-Resolution Using Very Deep Convolutional Networks """
    @with_config
    def __init__(self, **kwargs):
        KNetSR.__init__(self, **kwargs)

    def _define_models(self):
        KNetSR._define_models(self)
        with tf.name_scope('upsampling'):
            ups = UpSampling2D(
                size=self._down_sample_ratios[self._nb_down_sample])(self._ipn)
        x = ups
        for i, nc in enumerate(self._hiddens):
            with tf.name_scope('conv_%d' % i):
                x = Conv2D(nc, 3, activation='elu', padding='same')(x)
        with tf.name_scope('output'):
            res_inf = Conv2D(1, 3, padding='same')(x)
            img_inf = add([res_inf, ups])
        with tf.name_scope('res_out'):
            res_out = sub(self._ips[0], img_inf)
        self._models[self.model_id('sr')] = Model(self._ipn, img_inf)
        self._models[self.model_id('res_out')] = Model(
            [self._ips[0], self._ipn], res_out)


class SRDv1b(KNetSR):
    """ based on arxiv Accurate Image Super-Resolution Using Very Deep Convolutional Networks, with batch norm. """
    @with_config
    def __init__(self, **kwargs):
        KNetSR.__init__(self, **kwargs)

    def _define_models(self):
        KNetSR._define_models(self)
        with tf.name_scope('upsampling'):
            ups = UpSampling2D(
                size=self._down_sample_ratios[self._nb_down_sample])(self._ipn)
        x = ups
        for i, nc in enumerate(self._hiddens):
            with tf.name_scope('conv_%d' % i):
                x = Conv2D(nc, 3, activation='elu', padding='same')(x)
                x = BatchNormalization()(x)
        with tf.name_scope('output'):
            res_inf = Conv2D(1, 3, padding='same')(x)
            img_inf = add([res_inf, ups])
        with tf.name_scope('res_out'):
            res_out = sub(self._ips[0], img_inf)
        self._models[self.model_id('sr')] = Model(self._ipn, img_inf)
        self._models[self.model_id('res_out')] = Model(
            [self._ips[0], self._ipn], res_out)


class SRDv2(KNetSR):
    """ UpSampling2D in the end"""
    @with_config
    def __init__(self, **kwargs):
        KNetSR.__init__(self, **kwargs)

    def _define_models(self):
        KNetSR._define_models(self)

        x = self._ipn
        for i, nc in enumerate(self._hiddens):
            with tf.name_scope('conv_%d' % i):
                x = Conv2D(nc, 3, activation='elu', padding='same')(x)
        with tf.name_scope('upsampling'):
            x = UpSampling2D(
                size=self._down_sample_ratios[self._nb_down_sample])(x)
        with tf.name_scope('conv_end'):
            x = Conv2D(self._hiddens[-1], 3, activation='elu', padding='same')(x)
        with tf.name_scope('output'):
            res_inf = Conv2D(1, 3, padding='same')(x)
            img_inf = add([res_inf, ups])
        with tf.name_scope('res_out'):
            res_out = sub(self._ips[0], img_inf)
        self._models[self.model_id('sr')] = Model(self._ipn, img_inf)
        self._models[self.model_id('res_out')] = Model(
            [self._ips[0], self._ipn], res_out)


class SRDMultiScale(KNetSR):
    @with_config
    def __init__(self,
                 nb_kernels=[64] * 20,
                 **kwargs):
        KNetSR.__init__(self, **kwargs)
        self._models_names = ['sr']
        for i in range(self._nb_down_sample):
            self._models_names.append(['sr_%d' % i])
        self._is_trainable = [True] * (self._nb_down_sample + 1)

    def _define_models(self):
        KNetSR._define_models(self)
        x = self._ipn
        for i_up in range(self._nb_down_sample):
            with tf.name_scope('up_%d' % i_up):
                x = UpSampling2D(size=self._down_sample_ratio)(self._ip)

        x = ups
        for i in range(20):
            with tf.name_scope('conv_%d' % i):
                x = Conv2D(64, 3, activation='elu', padding='same')(x)
                x = BatchNormalization()(x)
        with tf.name_scope('output'):
            res_inf = Conv2D(1, 3, padding='same')(x)
            img_inf = add([res_inf, ups])
        with tf.name_scope('res_out'):
            res_out = sub(self._ips[0], img_inf)
        self._models[self.model_id('sr')] = Model(self._ipn, img_inf)
        self._models[self.model_id('res_out')] = Model(
            [self._ips[0], self._ipn], res_out)

# class SRSCAAE(KNetSR):
#     """ Super resolution based on conditional adverserial autoencoders. """
#     @with_config
#     def __init__(self,
#                  **kwargs):
#         KNetSR.__init__(self, **kwargs)

#     def _define_models(self):
#         KNetSR._define_models(self)
#         with tf.name_scope('upsampling'):
#             ups = UpSampling2D(
#                 size=self._down_sample_ratios[self._nb_down_sample])(self._ipn)
#         x = ups
#         for i in range(20):
#             with tf.name_scope('conv_%d' % i):
#                 x = Conv2D(64, 3, activation='elu', padding='same')(x)
#                 x = BatchNormalization()(x)
#         with tf.name_scope('output'):
#             res_inf = Conv2D(1, 3, padding='same')(x)
#             img_inf = add([res_inf, ups])
#         with tf.name_scope('res_out'):
#             res_out = sub(self._ips[0], img_inf)
#         self._models[self.model_id('sr')] = Model(self._ipn, img_inf)
#         self._models[self.model_id('res_out')] = Model(
#             [self._ips[0], self._ipn], res_out)

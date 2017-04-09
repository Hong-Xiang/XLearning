import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, ELU, LeakyReLU, Conv2D, UpSampling2D, BatchNormalization, Cropping2D, add, Lambda, Cropping2D

from keras import backend as K
from keras.engine.topology import Layer


import xlearn.utils.xpipes as utp
from ..nets.base import Net
from ..utils.general import with_config, enter_debug
from ..utils.tensor import upsample_shape, downsample_shape
from ..models.merge import sub
from ..models.image import conv_blocks, conv_f, sr_base


IMAGE_SUMMARY_MAX_OUTPUT = 5
MAX_DOWNSAMPLE = 3


class NetSR(Net):
    @with_config
    def __init__(self,
                 is_down_sample_0=True,
                 is_down_sample_1=True,
                 nb_down_sample=MAX_DOWNSAMPLE,
                 is_deconv=False,
                 crop_size=None,
                 **kwargs):
        super(NetSR, self).__init__(**kwargs)

        self._is_down_sample_0 = is_down_sample_0
        self._is_down_sample_1 = is_down_sample_1
        self._nb_down_sample = nb_down_sample

        if crop_size is None:
            crop_size = (0, 0)
        self._crop_size = crop_size

        self._is_deconv = is_deconv

        self._res_inf = None
        self._res_ref = None
        self._models_names = ['sr', 'itp', 'res_itp', 'res_out']
        self._is_trainable = [True, False, False, False]
        self._is_save = [True, False, False, False]

        self._down_sample_ratio = [1, 1]
        if self._is_down_sample_0:
            self._down_sample_ratio[0] = 2
        if self._is_down_sample_1:
            self._down_sample_ratio[1] = 2
        self._down_sample_ratio = tuple(self._down_sample_ratio)
        self._shapes = []
        self._shapes.append(self._inputs_shapes[0])

        self._down_sample_ratios = []
        self._down_sample_ratios.append([1, 1])

        for i in range(self._nb_down_sample):
            self._shapes.append(downsample_shape(self._shapes[i],
                                                 list(self._down_sample_ratio) + [1]))
            self._down_sample_ratios.append(upsample_shape(self._down_sample_ratios[i],
                                                           self._down_sample_ratio))
        self._shapes_cropped = []
        for i, s in enumerate(self._shapes):
            shape_new = [self._shapes[i][0] - 2 * self._crop_size[0],
                         self._shapes[i][1] - 2 * self._crop_size[1],
                         self._shapes[i][2]]
            self._shapes_cropped.append(shape_new)

    def _define_models(self):
        self._ips = []
        for i in range(self._nb_down_sample + 1):
            with tf.name_scope('input_%dx' % (2**i)) as scope:
                self._ips.append(Input(self._shapes[i]))
        self._ipn = self._ips[self._nb_down_sample]
        self._ip0 = self._ips[0]
        # with tf.name_scope('input') as scope:
        #     self._ipn = Input(self._shapes[-1])
        #     self._ip0 = Input(self._shapes[0])
        # self._ups = []
        # for i in range(self._nb_down_sample):
        #     with tf.name_scope('upsample_%dx' % (2**(i + 1))):
        #         self._ups.append(UpSampling2D(
        #             size=self._down_sample_ratio)(self._ips[i + 1]))

        # # cropped residual references
        # self._rrs = []
        # for i in range(self._nb_down_sample):
        #     with tf.name_scope('residual_reference_%dx' % (2**(i + 1))):
        #         self._rrs.append(sub(self._ips[i], self._ups[i]))

        # self._ups = []
        with tf.name_scope('ip_c'):
            self._ip0_c = Cropping2D(self._crop_size)(self._ip0)

        with tf.name_scope('interpolation'):
            itp = UpSampling2D(size=self._down_sample_ratios[-1])(self._ipn)
            self._itp = Cropping2D(self._crop_size)(itp)
        self._models[self.model_id('itp')] = Model(
            [self._ip0, self._ipn], [self._itp, self._ip0_c])

        with tf.name_scope('res_itp'):
            #     res_itp = []
            #     for i in range(self._nb_down_sample + 1):
            #         res_itp.append(sub(self._ips[0], self._ups[i]))
            # self._models[self.model_id('res_itp')] = Model(self._ips, res_itp)
            self._res_itp = sub(self._ip0_c, self._itp)
        self._models[self.model_id('res_itp')] = Model(
            [self._ip0, self._ipn], self._res_itp)

    def dummy_label(self, model_id):
        pass

    @property
    def nb_down_sample(self):
        return self._nb_down_sample

    @property
    def crop_size(self):
        return self._crop_size

    def _train_model_on_batch(self, model_id, inputs, outputs):
        cpx, cpy = self.crop_size
        if cpx > 0 and cpy > 0:
            loss_v = self.model(model_id).train_on_batch(
                inputs[self._nb_down_sample], outputs[0][:, cpx:-cpx, cpy:-cpy, :])
        elif cpy > 0:
            loss_v = self.model(model_id).train_on_batch(
                inputs[self._nb_down_sample], outputs[0][:, :, cpy:-cpy, :])
        elif cpx > 0:
            loss_v = self.model(model_id).train_on_batch(
                inputs[self._nb_down_sample], outputs[0][:, cpx:-cpx, :, :])
        else:
            loss_v = self.model(model_id).train_on_batch(
                inputs[self._nb_down_sample], outputs[0])
        return loss_v

    def _predict(self, model_id, inputs):
        if model_id == 'sr':
            return self.model(model_id).predict(inputs[self._nb_down_sample], batch_size=self._batch_size)
        elif model_id == 'itp' or model_id == 'res_itp' or model_id == 'res_out':
            ips = [inputs[0], inputs[self._nb_down_sample]]
            return self.model(model_id).predict(ips, batch_size=self._batch_size)


class SRInterp(NetSR):
    @with_config
    def __init__(self, **kwargs):
        super(SRInterp, self).__init__(**kwargs)

    def image_resize(self, x):
        return tf.image.resize_bicubic(x, size=self._shapes[0][0:2])

    def image_output_shape(self, input_shape):
        return [None] + list(self._shapes[0])

    def ly_resize(self):
        return Lambda(function=self.image_resize, output_shape=self.image_output_shape)

    def _define_models(self):
        super(SRInterp, self)._define_models()
        with tf.name_scope('inference'):
            upl = self.ly_resize()(self._ipn)
            self._output = Cropping2D(self._crop_size)(upl)
        with tf.name_scope('res_out'):
            res_out = sub(self._ip0_c, self._output)
        self._is_trainable = [False, False, False]
        self._models[self.model_id('sr')] = Model(self._ipn, self._output)
        self._models[self.model_id('res_out')] = Model(
            [self._ip0, self._ipn], res_out)


class SRDv0(NetSR):
    """ based on arxiv 1501.00092 """
    @with_config
    def __init__(self, **kwargs):
        NetSR.__init__(self, **kwargs)

    def _define_models(self):
        NetSR._define_models(self)
        with tf.name_scope('upsampling'):
            ups = UpSampling2D(
                size=self._down_sample_ratios[self._nb_down_sample])(self._ipn)
        with tf.name_scope('conv_0'):
            x = Conv2D(64, 9, padding='same')(ups)
            if self._is_bn:
                x = BatchNormalization()(x)
            x = ELU()(x)
        with tf.name_scope('conv_1'):
            x = Conv2D(32, 1, padding='same')(x)
            if self._is_bn:
                x = BatchNormalization()(x)
            x = ELU()(x)
        with tf.name_scope('output'):
            res_inf = Conv2D(1, 5, padding='same')(x)
            img_inf = add([res_inf, ups])
            img_crop = Cropping2D(self._crop_size)(img_inf)
        with tf.name_scope('res_out'):
            res_out = sub(self._ip0_c, img_crop)
        self._models[self.model_id('sr')] = Model(self._ipn, img_crop)
        self._models[self.model_id('res_out')] = Model(
            [self._ip0, self._ipn], res_out)


class SRDv1(NetSR):
    """ based on arxiv Accurate Image Super-Resolution Using Very Deep Convolutional Networks """
    @with_config
    def __init__(self, **kwargs):
        NetSR.__init__(self, **kwargs)

    def _define_models(self):
        NetSR._define_models(self)
        with tf.name_scope('upsampling'):
            ups = UpSampling2D(
                size=self._down_sample_ratios[self._nb_down_sample])(self._ipn)
        x = ups
        for i, nc in enumerate(self._hiddens):
            with tf.name_scope('conv_%d' % i):
                x = Conv2D(nc, 3, padding='same')(x)
                if self._is_bn:
                    x = BatchNormalization()(x)
                x = ELU()(x)
        with tf.name_scope('output'):
            res_inf = Conv2D(1, 5, padding='same')(x)
            img_inf = add([res_inf, ups])
            img_crop = Cropping2D(self._crop_size)(img_inf)
        with tf.name_scope('res_out'):
            res_out = sub(self._ip0_c, img_crop)
        self._models[self.model_id('sr')] = Model(self._ipn, img_crop)
        self._models[self.model_id('res_out')] = Model(
            [self._ip0, self._ipn], res_out)


class SRDv2(NetSR):
    """ UpSampling2D in the end"""
    @with_config
    def __init__(self,
                 h_pre,
                 h_post,
                 **kwargs):
        NetSR.__init__(self, **kwargs)
        self._h_pre = h_pre
        self._h_post = h_post

    def _define_models(self):
        NetSR._define_models(self)
        with tf.name_scope('upsampling_ip'):
            ups = UpSampling2D(
                size=self._down_sample_ratios[self._nb_down_sample])(self._ipn)
        x = self._ipn
        for i, nc in enumerate(self._h_pre):
            with tf.name_scope('conv_pre_%d' % i):
                x = Conv2D(nc, 3, padding='same')(x)
                if self._is_bn:
                    x = BatchNormalization()(x)
                x = ELU()(x)
        with tf.name_scope('upsampling'):
            x = UpSampling2D(
                size=self._down_sample_ratios[self._nb_down_sample])(x)
        for i, nc in enumerate(self._h_post):
            with tf.name_scope('conv_post_%d' % i):
                x = Conv2D(nc, 3, padding='same')(x)
                if self._is_bn:
                    x = BatchNormalization()(x)
                x = ELU()(x)
        with tf.name_scope('output'):
            res_inf = Conv2D(1, 3, padding='same')(x)
            img_inf = add([res_inf, ups])
            img_crop = Cropping2D(self._crop_size)(img_inf)
        with tf.name_scope('res_out'):
            res_out = sub(self._ip0_c, img_crop)
        self._models[self.model_id('sr')] = Model(self._ipn, img_crop)
        self._models[self.model_id('res_out')] = Model(
            [self._ip0, self._ipn], res_out)


class SRDv3(NetSR):
    """ UpSampling2D in the end"""
    @with_config
    def __init__(self,
                 **kwargs):
        NetSR.__init__(self, **kwargs)

    def _define_models(self):
        NetSR._define_models(self)
        with tf.name_scope('upsampling_ip'):
            ups = UpSampling2D(
                size=self._down_sample_ratios[self._nb_down_sample])(self._ipn)
        x = self._ipn
        x = sr_base(x, self._down_sample_ratio[0], self._down_sample_ratio[1])
        with tf.name_scope('output'):
            res_inf = Conv2D(1, 3, padding='same')(x)
            img_inf = add([res_inf, ups])
            img_crop = Cropping2D(self._crop_size)(img_inf)
        with tf.name_scope('res_out'):
            res_out = sub(self._ip0_c, img_crop)
        self._models[self.model_id('sr')] = Model(self._ipn, img_crop)
        self._models[self.model_id('res_out')] = Model(
            [self._ip0, self._ipn], res_out)


class SRDv4(NetSR):
    @with_config
    def __init__(self,
                 **kwargs):
        NetSR.__init__(self, **kwargs)

    def _define_models(self):
        NetSR._define_models(self)


class SRAEv0(NetSR):
    """ Super resolution based on autoencoder"""
    @with_config
    def __init__(self,
                 latent_dim,
                 h_enc,
                 h_enc_cond,
                 h_dec,
                 h_dec_cond,
                 **kwargs):
        NetSR.__init__(self, **kwargs)
        self._h_enc = h_enc
        self._h_enc_cond = h_enc_cond
        self._h_dec = h_dec
        self._h_dec_cond = h_dec_cond

    def _define_models(self):
        NetSR._define_models(self)

        with tf.name_scope('encoder_hr'):
            x = self._ip0
            for i, nc in enumerate(self._h_enc):
                with tf.name_scope('conv_%d' % i):
                    x = Conv2D(nc, 3, padding='same')(x)
                    if self._is_bn:
                        x = BatchNormalization()(x)
                    x = ELU()(x)
            with tf.name_scope('conv_enc'):
                x = Conv2D(latent_dim, 3, padding='same')(x)
        with tf.name_scope('encoder_cond'):
            x = self._ipn
            for i, nc in enumerate(self._h_enc):
                with tf.name_scope('conv_%d' % i):
                    x = Conv2D(nc, 3, padding='same')(x)
                    if self._is_bn:
                        x = BatchNormalization()(x)
                    x = ELU()(x)
            with tf.name_scope('conv_enc'):
                x = Conv2D(latent_dim, 3, padding='same')(x)
        with tf.name_scope('upsampling'):
            x = UpSampling2D(
                size=self._down_sample_ratios[self._nb_down_sample])(x)
        with tf.name_scope('conv_end'):
            x = Conv2D(self._hiddens[-1], 5,
                       activation='elu', padding='same')(x)
            x = Conv2D(self._hiddens[-1], 5,
                       activation='elu', padding='same')(x)
        with tf.name_scope('output'):
            res_inf = Conv2D(1, 3, padding='same')(x)
            img_inf = add([res_inf, ups])
            img_crop = Cropping2D(self._crop_size)(img_inf)
        with tf.name_scope('res_out'):
            res_out = sub(self._ip0_c, img_inf)
        self._models[self.model_id('sr')] = Model(self._ipn, img_crop)
        self._models[self.model_id('res_out')] = Model(
            [self._ip0, self._ipn], res_out)

# class SRDMultiScale(NetSR):
#     @with_config
#     def __init__(self,
#                  nb_kernels=[64] * 20,
#                  **kwargs):
#         NetSR.__init__(self, **kwargs)
#         self._models_names = ['sr']
#         for i in range(self._nb_down_sample):
#             self._models_names.append(['sr_%d' % i])
#         self._is_trainable = [True] * (self._nb_down_sample + 1)

#     def _define_models(self):
#         NetSR._define_models(self)
#         x = self._ipn
#         for i_up in range(self._nb_down_sample):
#             with tf.name_scope('up_%d' % i_up):
#                 x = UpSampling2D(size=self._down_sample_ratio)(self._ip)

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

# class SRSCAAE(NetSR):
#     """ Super resolution based on conditional adverserial autoencoders. """
#     @with_config
#     def __init__(self,
#                  **kwargs):
#         NetSR.__init__(self, **kwargs)

#     def _define_models(self):
#         NetSR._define_models(self)
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

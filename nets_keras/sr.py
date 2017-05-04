import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, ELU, LeakyReLU, Conv2D, UpSampling2D, BatchNormalization, Cropping2D, add, Lambda, Cropping2D, ZeroPadding2D, MaxPool2D, concatenate

from keras import backend as K
from keras.engine.topology import Layer


import xlearn.utils.xpipes as utp
from ..nets.base import Net
from ..utils.general import with_config, enter_debug
from ..utils.tensor import upsample_shape, downsample_shape
from ..models.merge import sub
from ..models.image import conv_blocks, conv_f, sr_base, sr_end


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

    def _scadule_model(self):
        return 'sr'

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
                 is_celu=False,
                 **kwargs):
        NetSR.__init__(self, **kwargs)
        self._is_celu = is_celu

    def _define_models(self):
        self._ips = []
        for i in range(self._nb_down_sample + 1):
            with tf.name_scope('input_%dx' % (2**i)) as scope:
                self._ips.append(Input(self._shapes[i]))
        self._ipn = self._ips[self._nb_down_sample]
        self._ip0 = self._ips[0]

        with tf.name_scope('upsampling_ip'):
            ups = UpSampling2D(
                size=self._down_sample_ratios[self._nb_down_sample])(self._ipn)
        x = ups
        # x = sr_base(x, self._down_sample_ratio[0], self._down_sample_ratio[1])
        mc = conv_blocks(x.shape.as_list()[
                         1:], self._hiddens, 3, is_celu=self._is_celu)
        with tf.name_scope('infer'):
            x = mc(x)
        with tf.name_scope('output'):
            res_inf = Conv2D(1, 3, padding='valid')(x)
            spo = res_inf.shape.as_list()[1:3]
            spi = ups.shape.as_list()[1:3]
            cpx = (spi[0] - spo[0]) // 2
            cpy = (spi[1] - spo[1]) // 2
            self._crop_size = (cpx, cpy)
            ups_c = Cropping2D(self._crop_size)(ups)
            img_inf = add([res_inf, ups_c])
            img_crop = img_inf
        with tf.name_scope('res_out'):
            with tf.name_scope('ip_c'):
                self._ip0_c = Cropping2D(self._crop_size)(self._ip0)
            res_out = sub(self._ip0_c, img_crop)
        self._models[self.model_id('sr')] = Model(self._ipn, img_crop)
        self._models[self.model_id('res_out')] = Model(
            [self._ip0, self._ipn], res_out)

        with tf.name_scope('interpolation'):
            itp = UpSampling2D(size=self._down_sample_ratios[-1])(self._ipn)
            self._itp = Cropping2D(self._crop_size)(itp)
        self._models[self.model_id('itp')] = Model(
            [self._ip0, self._ipn], [self._itp, self._ip0_c])

        with tf.name_scope('res_itp'):
            self._res_itp = sub(self._ip0_c, self._itp)
        self._models[self.model_id('res_itp')] = Model(
            [self._ip0, self._ipn], self._res_itp)


class SRDv4(NetSR):
    @with_config
    def __init__(self,
                 nb_phase=32,
                 **kwargs):
        NetSR.__init__(self, **kwargs)
        self._models_names = ['sr',  'sr1', 'sr2', 'sr3',
                              'res_out', 'res_out_1', 'res_out_2', 'res_out_3',
                              'res_itp', 'res_itp_1', 'res_itp_2', 'res_itp_3',
                              'itp', 'srdebug']
        self._is_trainable = [True, True, True, True,
                              False, False, False, False,
                              False, False, False, False,
                              False, False]
        self._is_save = [True, False, False, False,
                         False, False, False, False,
                         False, False, False, False,
                         False, False]
        self._nb_phase = nb_phase

    def _define_models(self):
        ips = []
        for i in range(self._nb_down_sample + 1):
            with tf.name_scope('input_%dx' % (2**i)) as scope:
                ips.append(Input(self._shapes[i], name='input_%dx' % i))
        ip3 = ips[3]
        ip2 = ips[2]
        ip1 = ips[1]
        ip0 = ips[0]

        with tf.name_scope('net_8x'):

            up3 = UpSampling2D(size=self._down_sample_ratio,
                               name='up_ip8x')(ip3)
            m3 = conv_blocks(ips[3].shape.as_list()[1:],
                             self._hiddens, 3, name='def_conv_8x')
            with tf.name_scope('conv_8x'):
                res3 = m3(up3)
            inf3, cs3, res_inf3, res_itp3 = sr_end(
                res3, up3, ips[2], name='outputs_8x')
            self._models[self.model_id('sr3')] = Model(ip3, inf3)
            self._models[self.model_id('res_out_3')] = Model([
                ip3, ip2], res_inf3)
            self._models[self.model_id('res_itp_3')] = Model([
                ip3, ip2], res_itp3)
            self._crop_size_4x = cs3

        with tf.name_scope('net_4x'):
            up2 = UpSampling2D(size=self._down_sample_ratio,
                               name='up_ip4x')(ip2)
            shape_4x_input = ip2.shape.as_list()
            shape_4x_cropped = [shape_4x_input[1] - 2 * self._crop_size_4x[0],
                                shape_4x_input[2] - 2 * self._crop_size_4x[1],
                                1]
            up2c = Cropping2D(self._crop_size_4x, name='up_4x_cropped')(up2)
            m2 = conv_blocks(shape_4x_cropped, self._hiddens,
                             3, name='def_conv_4x')
            with tf.name_scope('conv_4x'):
                res2 = m2(up2c)
            inf2, cs2, res_inf2, res_itp2 = sr_end(
                res2, up2, ips[1], name='outputs_4x')
            self._models[self.model_id('sr2')] = Model(ip2, inf2)
            self._models[self.model_id('res_out_2')] = Model([
                ip2, ip1], res_inf2)
            self._models[self.model_id('res_itp_2')] = Model([
                ip2, ip1], res_itp2)
            self._crop_size_2x = cs2

        with tf.name_scope('net_2x'):
            up1 = UpSampling2D(size=self._down_sample_ratio,
                               name='up_ip2x')(ip1)
            up1c = Cropping2D(self._crop_size_2x, name='up_4x_cropped')(up1)
            shape_2x_input = ip1.shape.as_list()
            shape_2x_cropped = [shape_2x_input[1] - 2 * self._crop_size_2x[0],
                                shape_2x_input[2] - 2 * self._crop_size_2x[1],
                                1]
            m1 = conv_blocks(shape_2x_cropped, self._hiddens,
                             3, name='def_conv_2x')
            with tf.name_scope('conv_2x'):
                res1 = m1(up1c)
            inf1, cs1, res_inf1, res_itp1 = sr_end(
                res1, up1, ips[0], name='outputs_2x')
            self._models[self.model_id('sr1')] = Model(ip1, inf1)
            self._models[self.model_id('res_out_1')] = Model([
                ip1, ip0], res_inf1)
            self._models[self.model_id('res_itp_1')] = Model([
                ip1, ip0], res_itp1)
            self._crop_size_1x = cs1

        self._crop_size = cs1
        with tf.name_scope('net_full'):
            with tf.name_scope('net_4x_to_2x'):
                with tf.name_scope('inf_4x_cropped'):
                    inf_4x = inf3
                    inf_4x_p = ZeroPadding2D(
                        self._crop_size_4x, name='pad_inf_4x')(inf_4x)
                    up_2x = UpSampling2D(
                        size=self._down_sample_ratio, name='up_inf_4x')(inf_4x_p)
                    up_2x_c = Cropping2D(
                        self._crop_size_4x, name='crop_4x')(up_2x)
                with tf.name_scope('conv_4x'):
                    res_2x = m2(up_2x_c)
            inf_2x, _, _, _ = sr_end(
                res_2x, up_2x, ip1, name='outputs_4x_to_2x', is_res=False)
            with tf.name_scope('net_2x_to_1x'):
                with tf.name_scope('inf_2x_cropped'):
                    inf_2x_p = ZeroPadding2D(
                        self._crop_size_2x, name='pad_inf_2x')(inf_2x)
                    up_1x = UpSampling2D(
                        size=self._down_sample_ratio, name='up_inf_2x')(inf_2x_p)
                    up_1x_c = Cropping2D(
                        self._crop_size_2x, name='crop_2x')(up_1x)
                with tf.name_scope('conv_2x'):
                    res_1x = m1(up_1x_c)
            inft, cst, res_inft, _ = sr_end(
                res_1x, up_1x, ip0, name='outputs_2x_to_1x')
        self._models[self.model_id('sr')] = Model(ip3, inft)
        self._models[self.model_id('res_out')] = Model([ip3, ip0], res_inft)
        self._models[self.model_id('srdebug')] = Model(
            [ip3, ip0], [inf_4x, inf_2x, inft])

        with tf.name_scope('res_itp'):
            with tf.name_scope('interpolation'):
                ipt_full = UpSampling2D(
                    size=self._down_sample_ratios[-1], name='up_full')(ip3)
                ipt_full_c = Cropping2D(
                    self._crop_size_1x, name='crop_full')(ipt_full)
            ip0_c = Cropping2D(self._crop_size_1x,
                               name='input_1x_cropped')(ips[0])
            with tf.name_scope('res_full'):
                res_ipt_f = sub(ip0_c, ipt_full_c)
        # self._models[self.model_id('itp')] = Model(ip3, ipt_full_c)
        self._models[self.model_id('res_itp')] = Model([ip3, ip0], res_ipt_f)
        self._models[self.model_id('itp')] = Model(
            [ip3, ip0], [ipt_full_c, ip0_c])

    def _train_model_on_batch(self, model_id, inputs, outputs):
        if model_id == 'sr1':
            cpx, cpy = self._crop_size_1x
            loss_v = self.model(model_id).train_on_batch(
                inputs[1], inputs[0][:, cpx:-cpx, cpy:-cpy, :])
        elif model_id == 'sr2':
            cpx, cpy = self._crop_size_2x
            loss_v = self.model(model_id).train_on_batch(
                inputs[2], inputs[1][:, cpx:-cpx, cpy:-cpy, :])
        elif model_id == 'sr3':
            cpx, cpy = self._crop_size_4x
            loss_v = self.model(model_id).train_on_batch(
                inputs[3], inputs[2][:, cpx:-cpx, cpy:-cpy, :])
        elif model_id == 'sr':
            cpx, cpy = self._crop_size_1x
            loss_v = self.model(model_id).train_on_batch(
                inputs[self._nb_down_sample], outputs[0][:, cpx:-cpx, cpy:-cpy, :])
        else:
            raise ValueError(
                'Invalid or non-trainable model id {0}.'.format(model_id))
        return loss_v

    def _scadule_model(self):
        id_phase = self.global_step // self._nb_phase
        if id_phase >= 3:
            return 'sr'
        else:
            return 'sr%d' % (id_phase + 1)

    def _predict(self, model_id, inputs):
        if model_id == 'sr3':
            return self.model(model_id).predict(inputs[3], batch_size=self._batch_size)
        elif model_id in ['res_out_3', 'res_itp_3']:
            return self.model(model_id).predict([inputs[3], inputs[2]], batch_size=self._batch_size)
        elif model_id == 'sr2':
            return self.model(model_id).predict(inputs[2], batch_size=self._batch_size)
        elif model_id in ['res_out_2', 'res_itp_2']:
            return self.model(model_id).predict([inputs[2], inputs[1]], batch_size=self._batch_size)
        elif model_id == 'sr1':
            return self.model(model_id).predict(inputs[1], batch_size=self._batch_size)
        elif model_id in ['res_out_1', 'res_itp_1']:
            return self.model(model_id).predict([inputs[1], inputs[0]], batch_size=self._batch_size)
        elif model_id in ['sr']:
            return self.model(model_id).predict(inputs[3], batch_size=self._batch_size)
        elif model_id in ['res_out', 'res_itp', 'itp', 'srdebug']:
            return self.model(model_id).predict([inputs[3], inputs[0]], batch_size=self._batch_size)


class SRCAEv0(NetSR):
    """ Super resolution based on autoencoder"""
    @with_config
    def __init__(self,
                 h_enc_h,
                 h_enc_c,
                 h_dec,
                 **kwargs):
        NetSR.__init__(self, **kwargs)
        self._h_enc_h = h_enc_h
        self._h_enc_c = h_enc_c
        self._h_dec = h_dec

    def _define_models(self):
        self._ips = []
        for i in range(self._nb_down_sample + 1):
            with tf.name_scope('input_%dx' % (2**i)) as scope:
                self._ips.append(Input(self._shapes[i]))
        self._ip3 = self._ips[3]
        self._ip2 = self._ips[2]
        self._ip0 = self._ips[0]

        with tf.name_scope('encoder_hr'):
            x = self._ip2
            enc_h_f = conv_blocks(x.shape.as_list()[1:], self._h_enc_h, 3, name='enc_h')(x)
            enc_h = MaxPool2D(self._down_sample_ratio)(enc_h_f)
        with tf.name_scope('encoder_cond'):
            x = self._ip3
            enc_c = conv_blocks(x.shape.as_list()[1:], self._h_enc_c, 3, name='enc_c')(x)
        with tf.name_scope('merging'):
            minx = min(enc_h.shape.as_list()[1], enc_c.shape.as_list()[1])
            miny = min(enc_h.shape.as_list()[2], enc_c.shape.as_list()[2])
            cpx_h = (enc_h.shape.as_list()[1] - minx) // 2
            cpx_c = (enc_c.shape.as_list()[1] - minx) // 2
            cpy_h = (enc_h.shape.as_list()[2] - miny) // 2
            cpy_c = (enc_c.shape.as_list()[2] - miny) // 2
            enc_h = Cropping2D((cpx_h, cpy_h))(enc_h)
            enc_c = Cropping2D((cpx_c, cpy_c))(enc_c)
        with tf.name_scope('upsampling'):
            enc_h_1x = UpSampling2D(size=self._down_sample_ratio)(enc_h)
            enc_c_1x = UpSampling2D(size=self._down_sample_ratio)(enc_c)
            z = concatenate([enc_h_1x, enc_c_1x])
        with tf.name_scope('dec'):
            reps = conv_blocks(z.shape.as_list()[
                               1:], self._h_dec, 3, is_final_active=True, name='dec')(z)
        with tf.name_scope('output'):
            res_inf = Conv2D(1, 3)(reps)
            shape_o = res_inf.shape.as_list()[1:3]
            shape_i = self._ip2.shape.as_list()[1:3]
            crop_x = (shape_i[0] - shape_o[0]) // 2
            crop_y = (shape_i[1] - shape_o[1]) // 2
            self._crop_size = (crop_x, crop_y)
            ups = UpSampling2D(size=self._down_sample_ratio)(self._ip3)
            ups_crop = Cropping2D(self._crop_size)(ups)
            img_inf = add([res_inf, ups_crop])
        with tf.name_scope('res_out'):
            ip2_c = Cropping2D(self._crop_size)(self._ip2)
            res_out = sub(ip2_c, img_inf)
        self._models[self.model_id('sr')] = Model([self._ip3, self._ip2], img_inf)
        self._models[self.model_id('res_out')] = Model(
            [self._ip3, self._ip2], res_out)

        self._models[self.model_id('itp')] = Model(
            [self._ip3, self._ip2], [ups_crop, ip2_c])

        with tf.name_scope('res_itp'):
            res_itp = sub(ip2_c, ups_crop)
        self._models[self.model_id('res_itp')] = Model(
            [self._ip3, self._ip2], res_itp)

    def _train_model_on_batch(self, model_id, inputs, outputs):
        if model_id == 'sr':
            cpx, cpy = self._crop_size
            loss_v = self.model(model_id).train_on_batch(
                [inputs[3], inputs[2]], inputs[2][:, cpx:-cpx, cpy:-cpy, :])
        else:
            raise ValueError(
                'Invalid or non-trainable model id {0}.'.format(model_id))
        return loss_v
    
    def _predict(self, model_id, inputs):        
        return self.model(model_id).predict([inputs[3], inputs[2]], batch_size=self._batch_size)
        

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

import logging
import tensorflow as tf
from keras.layers import Input, Convolution2D, BatchNormalization, ELU, merge, UpSampling2D, Deconvolution2D

from ..model.layers import Input, Label, Convolution2DwithBN
from ..utils.general import with_config
from ..utils.tensor import upsample_shape
from .base import Net

IMAGE_SUMMARY_MAX_OUTPUT = 5


class SRNetBase(Net):
    """Base net of super resolution nets
    """
    @with_config
    def __init__(self,
                 shape_i=None,
                 shape_o=None,
                 down_sample_ratio=None,
                 is_deconv=False,
                 settings=None,
                 **kwargs):
        super(SRNetBase, self).__init__(**kwargs)
        self._settings = settings
        logging.getLogger(__name__).info(
            "=" * 8 + "SuperNet." + SRNetBase.__name__ + " Constructing." + "=" * 8)
        self._models_names = ['SuperResolution']
        self._is_train_step = [True]

        # Gather settings
        self._shape_i = self._update_settings('shape_i', shape_i)
        self._shape_o = self._update_settings('shape_o', shape_o)
        self._down_sample_ratio = self._update_settings(
            'down_sample_ratio', down_sample_ratio)
        self._is_deconv = self._update_settings('is_deconv', is_deconv)

        # Check settings
        shape_o_cal = upsample_shape(self._shape_i, self._down_sample_ratio)
        shape_o_cal = tuple(shape_o_cal)
        self._shape_o = tuple(self._shape_o)
        if shape_o_cal != self._shape_o:
            raise ValueError('Inconsistant shape_i, shape_o and ratio.')

        self._residual_ref = None

    def _define_losses(self):
        with tf.name_scope('loss'):
            loss_residual = tf.losses.mean_squared_error(
                self._residual_ref, self._residual_inf)
        self._losses[0] = loss_residual
        tf.summary.scalar(name='residual_mse_loss', tensor=loss_residual)

    def _define_models(self):
        with tf.name_scope('low_res_input'):
            low_res = Input(shape=self._shape_i, name='low_res')
        with tf.name_scope('high_res_lable'):
            high_res = Label(shape=self._shape_o, name="high_res")
        with tf.name_scope('upsampling'):
            interp = tf.image.resize_images(low_res, self._shape_o[:2])
        with tf.name_scope('residual_refernce'):
            residual_ref = tf.subtract(high_res, interp, name='residual_sub')
            self._residual_ref = residual_ref
        self._residual_inf = None
        self._inputs[0] = [low_res]
        self._labels[0] = [high_res]
        self._interp = interp

        tf.summary.image('low_resolution', low_res,
                         max_outputs=IMAGE_SUMMARY_MAX_OUTPUT)
        tf.summary.image('high_resolution', high_res,
                         max_outputs=IMAGE_SUMMARY_MAX_OUTPUT)
        tf.summary.image('residual_reference', residual_ref,
                         max_outputs=IMAGE_SUMMARY_MAX_OUTPUT)
        tf.summary.image('interpolation', interp,
                         max_outputs=IMAGE_SUMMARY_MAX_OUTPUT)
        logging.getLogger(__name__).info(
            "=" * 8 + "SuperNet." + SRNetBase.__name__ + " Constructed." + "=" * 8)

    @property
    def shape_i(self):
        return self._shape_i

    @property
    def shape_o(self):
        return self._shape_o


class SRNetInterp(SRNetBase):

    @with_config
    def __init__(self, settings=None, **kwargs):
        super(SRNetInterp, self).__init__(**kwargs)
        self._is_train_step = [False]
        self._dummy_variable = tf.Variable([0.0])

    def _define_models(self):
        super(SRNetInterp, self)._define_models()
        infer = tf.image.resize_images(self._inputs[0][0], self._shape_o[:2])
        with tf.name_scope('inference'):
            self._residual_inf = tf.zeros(
                shape=[self._batch_size] + list(self._shape_o), dtype=tf.float32, name='zeros')
            self._outputs[0] = tf.add(
                infer, self._residual_inf, name='add_residual')


class SRSimple(SRNetBase):

    @with_config
    def __init__(self, settings=None, **kwargs):
        super(SRSimple, self).__init__(**kwargs)

    def _define_models(self):
        super(SRSimple, self)._define_models()
        with tf.name_scope('conv0'):
            conv0 = Convolution2D(64, 9, 9, border_mode='same',
                                  activation='relu', name='conv0')(self._interp)
        with tf.name_scope('conv_fc'):
            convf = Convolution2D(32, 1, 1, border_mode='same',
                                  activation='relu', name='dense_conv')(conv0)
        with tf.name_scope('conv_infer'):
            infer = Convolution2D(
                1, 5, 5, border_mode='same', name='conv2')(convf)
        self._residual_inf = infer
        tf.summary.image("residual_inference", infer,
                         max_outputs=IMAGE_SUMMARY_MAX_OUTPUT)
        with tf.name_scope('high_res_inference'):
            self._outputs[0] = [
                tf.add(infer, self._interp, name='add_residual')]
        tf.summary.image("superresolution_inference", self._outputs[
                         0][0], max_outputs=IMAGE_SUMMARY_MAX_OUTPUT)


class SRClassic(SRNetBase):

    @with_config
    def __init__(self, settings=None, **kwargs):
        super(SRClassic, self).__init__(**kwargs)

    def _define_models(self):
        super(SRClassic, self)._define_models()
        conv = self._interp
        for nb_f in range(self._hiddens):
            conv = Convolution2D(
                nb_f, 3, 3, border_mode='same', activation='elu')(conv)
        infer = Convolution2D(1, 3, 3, border_mode='same')(conv)
        self._residual_inf = infer
        tf.summary.image("residual_inference", infer,
                         max_outputs=IMAGE_SUMMARY_MAX_OUTPUT)
        self._outputs[0] = [tf.add(infer, self._interp, name='add_residual')]
        tf.summary.image("superresolution_inference", self._outputs[1][0],
                         max_outputs=IMAGE_SUMMARY_MAX_OUTPUT)


class SRF3D(SRNetBase):

    @with_config
    def __init__(self,
                 nb_deconv_filters=64,
                 nb_input_filters=64,
                 nb_input_row=3,
                 nb_res_blocks=2,
                 settings=None, **kwargs):
        super(SRF3D, self).__init__(**kwargs)

        # arch:
        #  1. default - with BN, ResNet on low res
        #  2. no_bn - without BN, ResNet on low res
        #  3. upsp - with BN, ResNet on high res
        #  4. no_bn_upsp - without BN, ResNet on high res
        self._settings = settings
        self._nb_deconv_filters = self._update_settings(
            'nb_deconv_filters', nb_deconv_filters)
        self._nb_input_filters = self._update_settings(
            'nb_input_filters', nb_input_filters)
        self._nb_input_row = self._update_settings(
            'nb_input_row', nb_input_row)
        self._nb_res_blocks = self._update_settings(
            'nb_res_block', nb_res_blocks)

    def __conv_block(self, tensor_in, nb_filter, name):
        with tf.name_scope(name):
            c = tensor_in
            for i in range(2):
                if self._arch == "default" or self._arch == "upsp":
                    c = Convolution2DwithBN(
                        c, nb_filter, 3, 3, name='CONV_BN_%d' % i)
                else:
                    c = Convolution2D(
                        nb_filter, 3, 3, activation='elu', border_mode='same')(c)
        return c

    def __res_block(self, tensor_in, nb_filters, name):
        with tf.name_scope(name):
            c0 = self.__conv_block(tensor_in, nb_filters, 'conv0')
            s0 = merge([c0, tensor_in], mode='sum')
            c1 = self.__conv_block(s0, nb_filters, 'conv1')
            s1 = merge([c1, s0], mode='sum')
            c2 = self.__conv_block(s1, nb_filters, 'conv2')
            s2 = merge([c2, s1], mode='concat')
        return s2

    def _define_models(self):
        super(SRF3D, self)._define_models()
        low_res = self._inputs[0][0]
        interp = self._interp
        high_res = self._labels[0][0]
        x = low_res
        if self._arch == "upsp" or self._arch == "no_bn_upsp":
            with tf.name_scope('upsamlping'):
                if self._is_deconv:
                    x = Deconvolution2D(self._nb_deconv_filters, 3, 3, output_shape=[
                                        None] + list(self._shape_o)[:-1] + [256], subsample=(3, 3), border_mode='same')(x)
                else:
                    x = UpSampling2D(size=self._down_sample_ratio[
                        :2], name='upsample')(x)
        with tf.name_scope('input_block'):

            x = Convolution2D(self._nb_input_filters,
                              self._nb_input_row,
                              self._nb_input_row,
                              border_mode='same',
                              name='CONV_in')(x)
            if self._arch == 'default' or self._arch == 'upsp':
                x = BatchNormalization(name='BN_in')(x)
            x = ELU(name='ELU_in')(x)

        for i in range(self._nb_res_blocks):
            x = self.__res_block(
                x, int(x.get_shape()[-1]), name='residual_block%d' % i)

        if self._arch == "default" or self._arch == "no_bn":
            with tf.name_scope('upsamlping'):
                if self._is_deconv:
                    x = Deconvolution2D(self._nb_deconv_filters, 3, 3, output_shape=[
                                        None] + list(self._shape_o)[:-1] + [x.get_shape[-1]], subsample=(3, 3), border_mode='same')(x)
                else:
                    x = UpSampling2D(size=self._down_sample_ratio[
                        :2], name='upsample')(x)                
        with tf.name_scope('conv_infer_0'):
            x = Convolution2D(512, 3, 3, border_mode='same',
                              activation='elu')(x)
        with tf.name_scope('residual_inference'):
            infer = Convolution2D(1, 5, 5, border_mode='same')(x)
        self._residual_inf = infer
        tf.summary.image("residual_inference", infer,
                         max_outputs=IMAGE_SUMMARY_MAX_OUTPUT)
        with tf.name_scope('high_residual_inference'):
            self._outputs[0] = [
                tf.add(infer, self._interp, name='add_residual')]
        tf.summary.image("superresolution_inference", self._outputs[0][0],
                         max_outputs=IMAGE_SUMMARY_MAX_OUTPUT)

import logging
import tensorflow as tf
from keras.layers import Input, Convolution2D, BatchNormalization, ELU, merge, UpSampling2D

from ..model.layers import Input, Label
from ..utils.general import with_config
from ..utils.tensor import upsample_shape
from .base import Net


class SRNetBase(Net):
    """Base net of super resolution nets
    """
    @with_config
    def __init__(self, filenames=None, settings=None, **kwargs):
        logging.getLogger(__name__).debug(
            'SRNetBase: filenames:{}'.format(filenames))
        super(SRNetBase, self).__init__(settings=settings, **kwargs)
        logging.getLogger(__name__).info(
            "=" * 8 + "SuperNet." + SRNetBase.__name__ + " Constructing." + "=" * 8)
        self._models_names = ['SuperResolutionResidual', 'SuperResolutionFull']
        self._is_train_step = [True, True]

        # Gather settings
        self._shape_i = settings['shape_i']
        self._c.update({'shape_i': self._shape_i})
        self._shape_o = settings['shape_o']
        self._c.update({'shape_o': self._shape_o})
        self._down_sample_ratio = settings['down_sample_ratio']
        self._c.update({'down_sample_ratio': self._down_sample_ratio})

        # Check settings
        shape_o_cal = upsample_shape(self._shape_i, self._down_sample_ratio)
        shape_o_cal = tuple(shape_o_cal)
        self._shape_o = tuple(self._shape_o)
        if shape_o_cal != self._shape_o:
            raise ValueError('Inconsistant shape_i, shape_o and ratio.')

    def _define_models(self):
        low_res = Input(shape=self._shape_i, name='low_res')
        high_res = Label(shape=self._shape_o, name="high_res")

        with tf.name_scope('residual_refernce'):
            interp = tf.image.resize_images(
                low_res, self._shape_o[:2])
            residual_ref = tf.subtract(high_res, interp, name='residual_sub')

        self._inputs[0] = [low_res]
        self._labels[0] = [residual_ref]
        self._inputs[1] = [low_res]
        self._labels[1] = [high_res]
        self._interp = interp

        tf.summary.image('low_resolution', low_res)
        tf.summary.image('high_resolution', high_res)
        tf.summary.image('residual_reference', residual_ref)
        tf.summary.image('interpolation', interp)
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
        super(SRNetInterp, self).__init__(self, settings=settings, **kwargs)
        self._is_train_step = [False, False]
        self._dummy_variable = tf.Variable([0.0])

    def _define_models(self):
        super(SRNetInterp, self)._define_models()
        infer = tf.image.resize_images(self._inputs[0][0], self._shape_o[:2])
        with tf.name_scope('inference'):
            self._outputs[0] = tf.zeros(
                shape=[self._batch_size] + list(self._shape_o), dtype=tf.float32, name='zeros')
            self._outputs[1] = tf.add(
                infer, self._outputs[0], name='add_residual')


class SRSimple(SRNetBase):

    @with_config
    def __init__(self, settings=None, **kwargs):
        super(SRSimple, self).__init__(self, **kwargs)

    def _define_models(self):
        super(SRSimple, self)._define_models()
        conv0 = Convolution2D(64, 9, 9, border_mode='same',
                              activation='relu', name='conv0')(self._interp)
        convf = Convolution2D(32, 1, 1, border_mode='same',
                              activation='relu', name='dense_conv')(conv0)
        infer = Convolution2D(1, 5, 5, border_mode='same', name='conv2')(convf)
        tf.summary.image("residual_inference", infer)
        self._outputs[0] = [infer]
        self._outputs[1] = [tf.add(infer, self._interp, name='add_residual')]

class SRClassic(SRNetBase):
    @with_config
    def __init__(self, settings=None, **kwargs):
        super(SRSimple, self).__init__(self, **kwargs)
    
    def _define_models(self):
        super(SRClassic, self)._define_models()
        conv = self._interp
        for nb_f in range(self._hiddens):
            conv = Convolution2D(nb_f, 3, 3, border_mode='same', activation='elu')(conv)
        infer = Convolution2D(1, 3, 3, border_mode='same')(conv)
        tf.summary.image("residual_inference", infer)
        self._outputs[0] = [infer]
        self._outputs[1] = [tf.add(infer, self._interp, name='add_residual')]
        tf.summary.image("superresolution_inference", self._outputs[1][0])

class SRF3D(SRNetBase):

    @with_config
    def __init__(self, settings=None, **kwargs):
        super(SRF3D, self).__init__(self, **kwargs)

    def __conv_block(self, tensor_in, nb_filters, name):
        with tf.name_scope(name):
            c0 = Convolution2D(
                nb_filters, 3, 3, border_mode='same', name='CONV0')(tensor_in)
            b0 = BatchNormalization(name='BN0')(c0)
            e0 = ELU(name='ELU0')(b0)
            c1 = Convolution2D(
                nb_filters, 3, 3, border_mode='same', name='CONV1')(tensor_in)
            b1 = BatchNormalization(name='BN1')(c0)
            e1 = ELU(name='ELU1')(b0)
        return e1

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
        high_res = self._labels[1][0]
        with tf.name_scope('input_block'):
            c_i = Convolution2D(64, 9, 9, border_mode='same',
                                name='CONV_in')(low_res)
            b_i = BatchNormalization(name='BN_in')(c_i)
            e_i = ELU(name='ELU_in')(b_i)

        res0 = self.__res_block(e_i, 64, name='residual_block0')
        res1 = self.__res_block(res0, 128, name='residual_block0')
        upsampled = UpSampling2D(size=(1, 4), name='upsample')(res1)
        e_o_0 = ELU(name='EO0')(upsampled)
        c_o_0 = Convolution2D(
            256, 3, 3, border_mode='same', name='CONV_O0')(e_o_0)
        e_o_1 = ELU(name='EO0')(c_o_0)
        infer = Convolution2D(1, 3, 3, border_mode='same',
                              name='CONV_O1_INFER')(e_o_1)
        tf.summary.image("residual_inference", infer)
        self._outputs[0] = [infer]
        self._outputs[1] = [tf.add(infer, self._interp, name='add_residual')]
        tf.summary.image("superresolution_inference", self._outputs[1][0])

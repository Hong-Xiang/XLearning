import logging
import tensorflow as tf
from keras.layers import Input

from ..model.layers import Input, Label
from ..utils.general import with_config
from ..utils.tensor import upsample_shape
from .base import Net


class SRNetBase(Net):
    """Base net of super resolution nets
    """
    @with_config
    def __init__(self, settings=None, **kwargs):
        logging.getLogger(__name__).debug(
            'SRNetBase: filenames:{}'.format(filenames))
        super(SRNetBase, self).__init__(settings=settings, **kwargs)
        logging.getLogger(__name__).info(
            "=" * 8 + "SuperNet." + SRNetBase.__name__ + " Constructing." + "=" * 8)
        self._models_names = ['SuperResolutionResidual', 'SuperResolutionFull']
        self._is_train_step = [True, False]

        # Gather settings
        self._shape_i = settings['shape_i']
        self._c.update({'shape_i': self._shape_i})
        self._shape_o = settings['shape_o']
        self._c.update({'shape_o': self._shape_o})
        self._down_sample_ratio = settings['down_sample_ratio']
        self._c.update({'down_sample_ratio': self._down_sample_ratio})

        # Check settings
        shape_o_cal = upsample_shape(self._shape_i, self._down_sample_ratio)
        if shape_o_cal != self._shape_o:
            raise ValueError('Inconsistant shape_i, shape_o and ratio.')

    def _define_models(self):
        low_res = Input(shape=self._shape_i, name='low_res')
        high_res = Label(shape=self._shape_o, name="high_res")

        with tf.name_scope('residual_refernce'):
            interp = tf.image.resize_images(
                low_res, self._shape_o[:2])
            residual_ref = tf.subtract(high_res, interp, name='residual_sub')

        self._inputs[0] = low_res
        self._labels[0] = residual_ref
        self._inputs[1] = low_res
        self._labels[1] = high_res

        tf.summary.image('low resolution', low_res)
        tf.summary.image('high resolution', high_res)
        tf.summary.image('residual ref', residual_ref)
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

    def _define_models(self):
        super(SRNetInterp, self)._define_models()
        infer = tf.image.resize_images(self._inputs[0], self._shape_o[:2])
        with tf.name_scope('inference'):
            self._outputs[0] = tf.zeros(
                shape=[self._batch_size] + list(self._shape_o), dtype=tf.float32, name='zeros')
            self._outputs[1] = tf.add(
                infer, self._outputs[0], name='add residual')

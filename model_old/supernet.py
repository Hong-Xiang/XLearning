"""
Super resolution nets.

Subnets must have following methods:
output = infer(input[s])
output = loss(input[s], lable[s])
variable_list = variables([flags])
"""
import logging
import tensorflow as tf
import xlearn.nets.layers as layer
import xlearn.nets.model as model
from xlearn.nets.model import TFNet

FLAGS = tf.app.flags.FLAGS
ACTIVITION_FUNCTION = tf.nn.elu


class SuperNetInterp(TFNet):
    """Interpolation 'net'
    """

    def __init__(self, filenames=None, name="SuperNetInterp", varscope=None, **kwargs):
        super(SuperNetInterp, self).__init__(
            filenames=filenames, name=name, varscope=varscope, **kwargs)
        self._is_skip_restore = True
        print("=" * 8 + "SuperNet." + name + " Constructed." + "=" * 8)
        self._input = layer.inputs([None,
                                    self._low_shape[0],
                                    self._low_shape[1],
                                    1],
                                   "input_low_res")
        self._infer = tf.image.resize_images(self._input,
                                             [self._high_shape[0],
                                              self._high_shape[1]])

    def _gather_paras(self):
        self._batch_size = self._paras['batch_size']
        self._down_sample_ratio = self._paras['down_sample_ratio']
        self._shape_i = self._paras['shape_i']
        self._shape_o = self._paras['shape_o']
        self._high_shape = [self._shape_o[0], self._shape_o[1]]
        self._low_shape = [self._shape_i[0], self._shape_i[1]]


class SuperNetBase(TFNet):
    """Base net of super resolution nets
    """

    def __init__(self,
                 filenames=None,
                 name='SuperNetBase',
                 varscope=tf.get_variable_scope(),
                 **kwargs):
        logging.getLogger(__name__).debug(
            'SuperNetBase: filenames:{}'.format(filenames))
        super(SuperNetBase, self).__init__(
            filenames=filenames, name=name, varscope=varscope, **kwargs)
        logging.getLogger(__name__).info(
            "=" * 8 + "SuperNet." + name + " Constructing." + "=" * 8)
        self._input = layer.inputs([None,
                                    self._shape_low[0],
                                    self._shape_low[1],
                                    1],
                                   "input_low_res")
        self._label = layer.labels([None,
                                    self._shape_high[0],
                                    self._shape_high[1],
                                    1],
                                   "input_high_res")
        with tf.name_scope('interpolation'):
            self._interp = tf.image.resize_images(self._input,
                                                  [self._shape_high[0],
                                                   self._shape_high[1]])
        with tf.name_scope('residual_reference'):
            self._residual_reference = tf.sub(self._label, self._interp,
                                              name='sub')

        self._residual_inference = None
        self._infer = None
        self._midops = []
        self._net_definition()
        self._loss = layer.loss_summation(name="train_loss")
        self._train = layer.trainstep_clip(
            self._loss, self._learn_rate, self._global_step)
        self._add_summary()
        logging.getLogger(__name__).info(
            "=" * 8 + "SuperNet." + name + " Constructed." + "=" * 8)
        logging.getLogger(__name__).info(self._flags())

    def _gather_paras(self):
        self._batch_size = self._paras['batch_size']
        self._down_sample_ratio = self._paras['down_sample_ratio']
        self._shape_i = self._paras['shape_i']
        self._shape_o = self._paras['shape_o']
        self._shape_high = [self._shape_o[0], self._shape_o[1]]
        self._shape_low = [self._shape_i[0], self._shape_i[1]]
        self._shape_batch_high = [self.batch_size,
                                  self.height_high, self.width_high, 0]
        self._shape_batch_low = [self.batch_size,
                                 self.height_low, self.width_low, 0]

    def _net_definition(self):
        with tf.name_scope('residual_inference'):
            self._residual_inference = tf.sub(
                self._label, self._interp, name='sub')
        with tf.name_scope('inference'):
            self._residual_inference = tf.add(self._interp, self._residual_inference,
                                              name='add')

    def _flags(self):
        infos = ''
        infos = infos + "=" * 6 + "Net flags:" + "=" * 6 + "\n"
        infos += "height_high\t{}\n".format(self.height_high)
        infos += "width_high\t{}\n".format(self.width_high)
        infos += "height_low\t{}\n".format(self.height_low)
        infos += "width_low\t{}\n".format(self.width_low)
        infos += "FLAGS.hidden_units\t{}\n".format(FLAGS.hidden_units)
        infos += "FLAGS.hidden_layer\t{}".format(FLAGS.hidden_layer)
        return infos

    def _add_summary(self):
        model.scalar_summary(self._loss)
        for opt in self._midops:
            model.activation_summary(opt)
        tf.summary.image('input_low', self._input)
        tf.summary.image('label_high', self._label)
        tf.summary.image('residual_inference', self._residual_inference)
        tf.summary.image('residual_reference', self._residual_reference)
        tf.summary.image('inference', self._infer)
        tf.summary.image('interp_result', self._interp)

    @property
    def height_high(self):
        return self._shape_high[0]

    @property
    def width_high(self):
        return self._shape_high[1]

    @property
    def height_low(self):
        return self._shape_low[0]

    @property
    def width_low(self):
        return self._shape_low[1]

    @property
    def batch_size(self):
        return self._batch_size


class SuperNet0(SuperNetBase):
    """
    Most naive implementation, based on https://arxiv.org/pdf/1501.00092.pdf
    """

    def __init__(self,
                 filenames=None,
                 name='SuperNet0',
                 varscope=tf.get_variable_scope(),
                 **kwargs):
        super(SuperNet0, self).__init__(filenames=filenames,
                                        name=name,
                                        varscope=varscope,
                                        **kwargs)

    def _net_definition(self):
        # self._batch_size = tf.get_shape(self._input)[0]

        conv0 = layer.conv_activate(self._interp, [9, 9, 1, 64],
                                    padding='SAME', name='conv0')

        fullc = layer.conv_activate(conv0, [1, 1, 64, 32],
                                    padding='SAME', name='fc')

        with tf.name_scope('residual_inference'):
            self._residual_inference = layer.convolution(fullc, [5, 5, 32, 1],
                                                         padding='SAME', name='residual_inference')

        # self._psnr = layer.psnr_loss(self._residual_inference, self._residual_reference, name='psnr_loss')
        loss = layer.l2_loss(
            self._residual_inference, self._residual_reference, name='l2_loss')

        self._infer = tf.add(
            self._interp, self._residual_inference, name='infer')


class SuperNet1(SuperNetBase):
    """
    Implementation based on ???
    """

    def __init__(self,
                 filenames=None,
                 name='SuperNet1',
                 varscope=None,
                 **kwargs):
        super(SuperNet1, self).__init__(filenames=filenames,
                                        name=name,
                                        varscope=varscope,
                                        **kwargs)

    def _net_definition(self):
        conv0 = layer.conv_activate(self._interp, [5, 5, 1, 128],
                                    padding='SAME', name='conv0',
                                    activation_function=ACTIVITION_FUNCTION)

        self._midops.append(conv0)
        conv1 = layer.conv_activate(conv0, [3, 3, 128, 128],
                                    padding='SAME', name='conv1',
                                    activation_function=ACTIVITION_FUNCTION)
        self._midops.append(conv1)
        conv2 = layer.conv_activate(conv1, [3, 3, 128, 128],
                                    padding='SAME', name='conv2',
                                    activation_function=ACTIVITION_FUNCTION)
        self._midops.append(conv2)
        conv3 = layer.conv_activate(conv2, [3, 3, 128, 128],
                                    padding='SAME', name='conv3',
                                    activation_function=ACTIVITION_FUNCTION)
        self._midops.append(conv3)
        fullc = layer.conv_activate(conv3, [1, 1, 128, 128],
                                    padding='SAME', name='fc',
                                    activation_function=ACTIVITION_FUNCTION)
        self._midops.append(fullc)
        reco3 = layer.conv_activate(fullc, [3, 3, 128, 32],
                                    padding='SAME', name='reco3',
                                    activation_function=ACTIVITION_FUNCTION)
        self._midops.append(reco3)

        with tf.name_scope('residual_inference'):
            self._residual_inference = layer.convolution(reco3, [3, 3, 32, 1],
                                                         padding='SAME', name='residual_inference')

        with tf.name_scope('inference'):
            self._residual_inference = tf.add(self._interp, self._residual_inference,
                                              name='add')

        # self._psnr = layer.psnr_loss(self._residual_inference, self._residual_reference, name='psnr_loss')
        self._l2_loss = layer.l2_loss(
            self._residual_inference, self._residual_reference, name='l2_loss')


class SuperNet2(SuperNetBase):
    """
    Implementation based on https://arxiv.org/pdf/1511.04587.pdf
    """

    def __init__(self,
                 filenames=None,
                 name='SuperNet2',
                 varscope=tf.get_variable_scope(),
                 **kwargs):
        logging.getLogger(__name__).debug(
            'SuperNet2: filenames:{}'.format(filenames))
        super(SuperNet2, self).__init__(filenames=filenames,
                                        name=name,
                                        varscope=varscope,
                                        **kwargs)

    def _net_definition(self):
        filter_shape = [3, 3, FLAGS.hidden_units, FLAGS.hidden_units]

        conv = layer.conv_activate(self._interp, [5, 5, 1, FLAGS.hidden_units],
                                   padding='SAME', name='conv_init',
                                   activation_function=ACTIVITION_FUNCTION)
        self._midops.append(conv)
        for i in range(FLAGS.hidden_layer):
            conv = layer.conv_activate(conv, filter_shape,
                                       padding='SAME', name='conv%d' % (i + 1),
                                       activation_function=ACTIVITION_FUNCTION)
            self._midops.append(conv)
        with tf.name_scope('residual_inference'):
            self._residual_inference = layer.convolution(conv, [3, 3, FLAGS.hidden_units, 1],
                                                         padding='SAME', name='residual_inference')

        # self._psnr = layer.psnr_loss(self._residual_inference, self._residual_reference, name='psnr_loss')
        self._l2_loss = layer.l2_loss(self._residual_inference,
                                      self._residual_reference, name='l2_loss')

        with tf.name_scope('inference'):
            self._infer = tf.add(self._interp,
                                 self._residual_inference, name='add')


class SuperNetCrop(SuperNetBase):
    """
    Reimplementation of cropping net.
    """

    def __init__(self,
                 filenames=None,
                 name='SuperNetCrop',
                 varscope=None,
                 **kwargs):
        super(SuperNetCrop, self).__init__(filenames=filenames,
                                           name=name,
                                           varscope=varscope,
                                           **kwargs)

    def _net_definition(self):
        preshape = self._shape_high
        postshape = [preshape[0] - FLAGS.hidden_layer *
                     2, preshape[1] - FLAGS.hidden_layer * 2]

        offset = [FLAGS.hidden_layer, FLAGS.hidden_layer]
        self._residual_crop = layer.crop(self._residual_reference,
                                         postshape, offset, num=FLAGS.batch_size)
        self._high_crop = layer.crop(
            self._label, postshape, offset, num=FLAGS.batch_size)

        self._midop = []
        filter_shape = [3, 3, FLAGS.hidden_units, FLAGS.hidden_units]
        conv = layer.conv_activate(self._interp, [5, 5, 1, FLAGS.hidden_units],
                                   padding='VALID', name='conv_init',
                                   activation_function=ACTIVITION_FUNCTION)
        self._midop.append(conv)
        for i in range(FLAGS.hidden_layer - 3):
            conv = layer.conv_activate(conv, filter_shape,
                                       padding='VALID', name='conv%d' % (i + 1),
                                       activation_function=ACTIVITION_FUNCTION)
            self._midops.append(conv)
        with tf.name_scope('residual_inference'):
            self._residual_inference = layer.convolution(conv, [3, 3, FLAGS.hidden_units, 1],
                                                         padding='VALID', name='residual_inference')

        # self._psnr = layer.psnr_loss(self._residual_inference, self._residual_reference, name='psnr_loss')

        self._l2_loss = layer.l2_loss(self._residual_inference,
                                      self._residual_crop,
                                      name='l2_loss')

        self._interp_crop = layer.crop(self._interp,
                                       postshape, offset, num=FLAGS.batch_size)

        self._infer = tf.add(self._interp_crop,
                             self._residual_inference, name='infer')

    def _add_summary(self):
        model.scalar_summary(self._loss)
        for opt in self._midops:
            model.activation_summary(opt)
        tf.summary.image('input_low', self._input)
        tf.summary.image('label_high', self._label)
        tf.summary.image('residual_inference', self._residual_inference)
        tf.summary.image('residual_reference', self._residual_crop)
        tf.summary.image('inference', self._infer)
        tf.summary.image('interp_crop', self._interp_crop)
        tf.summary.image('high_crop', self._high_crop)

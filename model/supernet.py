"""
MNIST nets.

Subnets must have following methods:
output = infer(input[s])
output = loss(input[s], lable[s])
variable_list = variables([flags])
"""
from __future__ import absolute_import, division, print_function
from six.moves import xrange
import tensorflow as tf
import xlearn.nets.layers as layer
import xlearn.nets.model as model
from xlearn.nets.model import TFNet

FLAGS = tf.app.flags.FLAGS
ACTIVITION_FUNCTION = tf.nn.relu


class SuperNetInterp(TFNet):

    def _net_definition(self, name="SuperNetInterp", varscope=tf.get_variable_scope()):
        super(SuperNetInterp, self).__init__(varscope=varscope)
        self._name = name
        self._ratio = FLAGS.down_ratio
        print("=" * 8 + "SuperNet." + name + " Constructed." + "=" * 8)
        self._input = layer.inputs([None,
                                    FLAGS.height,
                                    FLAGS.width,
                                    1],
                                   "input_low_res")
        self._width_only = FLAGS.only_down_width
        high_shape = self._high_shape()
        self._infer = tf.image.resize_images(self._input,
                                             high_shape[0],
                                             high_shape[1])

    def _high_shape(self):
        if self._width_only:
            high_shape = [FLAGS.height, FLAGS.width * self._ratio]
        else:
            high_shape = [FLAGS.height * self._ratio,
                          FLAGS.width * self._ratio]
        return high_shape


class SuperNetBase(TFNet):

    def __init__(self,
                 name='SuperNetBase',
                 varscope=tf.get_variable_scope()):
        super(SuperNetBase, self).__init__(varscope=varscope)
        self._name = name
        self._ratio = FLAGS.down_ratio
        print("=" * 8 + "SuperNet." + name + " Constructed." + "=" * 8)
        self._input = layer.inputs([None,
                                    FLAGS.height,
                                    FLAGS.width,
                                    1],
                                   "input_low_res")
        self._width_only = FLAGS.only_down_width
        high_shape = self._high_shape()
        self._label = layer.labels([None,
                                    high_shape[0],
                                    high_shape[1],
                                    1],
                                   "input_high_res")
        with tf.name_scope('interpolation'):
            self._interp = tf.image.resize_images(self._input,
                                                  high_shape[0],
                                                  high_shape[1])
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
        self._print_flags()

    def _high_shape(self):
        if self._width_only:
            high_shape = [FLAGS.height, FLAGS.width * self._ratio]
        else:
            high_shape = [FLAGS.height * self._ratio,
                          FLAGS.width * self._ratio]
        return high_shape

    def _net_definition(self):
        with tf.name_scope('residual_inference'):
            self._residual_inference = tf.sub(
                self._label, self._interp, name='sub')
        with tf.name_scope('inference'):
            self._residual_inference = tf.add(self._interp, self._residual_inference,
                                              name='add')

    def _print_flags(self):
        print("=" * 6 + "Net flags:" + "=" * 6)
        print("FLAGS.height\t", FLAGS.height)
        print("FLAGS.width\t", FLAGS.width)
        print("FLAGS.hidden_units\t", FLAGS.hidden_units)
        print("FLAGS.hidden_layer\t", FLAGS.hidden_layer)

    def _add_summary(self):
        model.scalar_summary(self._loss)
        for opt in self._midops:
            model.activation_summary(opt)
        tf.image_summary('input_low', self._input)
        tf.image_summary('label_high', self._label)
        tf.image_summary('residual_inference', self._residual_inference)
        tf.image_summary('residual_reference', self._residual_reference)
        tf.image_summary('inference', self._infer)
        tf.image_summary('interp_result', self._interp)


class SuperNet0(SuperNetBase):
    """
    Most naive implementation, based on https://arxiv.org/pdf/1501.00092.pdf
    """

    def __init__(self,
                 name='SuperNet0',
                 varscope=tf.get_variable_scope()):
        super(SuperNet0, self).__init__(name=name,
                                        varscope=varscope)

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
        self._l2_loss = layer.psnr_loss(
            self._residual_inference, self._residual_reference, name='l2_loss')

        self._infer = tf.add(
            self._interp, self._residual_inference, name='infer')


class SuperNet1(SuperNetBase):
    """
    Implementation based on ???
    """

    def __init__(self,
                 name='SuperNet1',
                 varscope=tf.get_variable_scope()):
        super(SuperNet1, self).__init__(varscope=varscope)

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
                 name='SuperNet2',
                 varscope=tf.get_variable_scope()):
        super(SuperNet2, self).__init__(name=name, varscope=varscope)

    def _net_definition(self):
        filter_shape = [3, 3, FLAGS.hidden_units, FLAGS.hidden_units]

        conv = layer.conv_activate(self._interp, [5, 5, 1, FLAGS.hidden_units],
                                   padding='SAME', name='conv_init',
                                   activation_function=ACTIVITION_FUNCTION)
        self._midops.append(conv)
        for i in xrange(FLAGS.hidden_layer):
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
                 name='SuperNetCrop',
                 varscope=tf.get_variable_scope()):
        super(SuperNetCrop, self).__init__(varscope=varscope)

    def _net_definition(self):
        preshape = self._high_shape()
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
        for i in xrange(FLAGS.hidden_layer - 3):
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
        tf.image_summary('input_low', self._input)
        tf.image_summary('label_high', self._label)
        tf.image_summary('residual_inference', self._residual_inference)
        tf.image_summary('residual_reference', self._residual_crop)
        tf.image_summary('inference', self._infer)
        tf.image_summary('interp_crop', self._interp_crop)
        tf.image_summary('high_crop', self._high_crop)


# class SuperNet(TFNet):
#     """Super resolution net
#     """
#     def __init__(self):
#         self._low_res_images = layer.input_layer([None,
#                                                   FLAGS.height,
#                                                   FLAGS.width,
#                                                   1],
#                                                  "input_low_res")

#         self._high_res_images = layer.label_layer([None,
#                                                   FLAGS.height,
#                                                   FLAGS.width,
#                                                   1],
#                                                   "input_high_res")

#         self._residual_cropped = layer.crop_layer(self._residual_full,
#                                                   [FLAGS.valid_h,
#                                                    FLAGS.valid_w],
#                                                   [FLAGS.valid_y,
#                                                    FLAGS.valid_x],
#                                                   name='crop_residual')


#         self._low_res_cropped = layer.crop_layer(self._low_res_images,
#                                                  [FLAGS.valid_h,
#                                                   FLAGS.valid_w],
#                                                  [FLAGS.valid_y,
#                                                   FLAGS.valid_x],
#                                                  'crop_low_resolution')

#         self._high_res_cropped = tf.add(self._low_res_cropped,
#                                         self._residual_cropped,
#                                         "label_high_res")

#         self._residual_inference = self._net_definition_deep_conv()

#         # self._residual_inference = self._net_definition_2016()

#         self._loss = layer.l2_loss_layer(self._residual_cropped,
#                                          self._residual_inference,
#                                          name="loss")

#         self._inference_image = tf.add(self._low_res_cropped,
#                                        self._residual_inference,
#                                        name="image_inference")

#         # Decay the learning rate exponentially based on the number of steps.
#         with tf.name_scope("train") as scope:
#             learn_rate = tf.train.exponential_decay(FLAGS.learning_rate_init,
#                                                     self._global_step,
#                                                     FLAGS.decay_steps,
#                                                     FLAGS.learning_rate_decay_factor,
#                                                     staircase=True,
#                                                     name=scope+'/learning_rate')

#             tf.scalar_summary("learning rate", learn_rate)

#             self._train_step = tf.train.AdamOptimizer(learn_rate).minimize(self._loss,
#                                                                            self._global_step,
# name=scope+'/train_step')

#             #self._train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step=step)

#         tf.image_summary('input_low', self._low_res_cropped)
#         tf.image_summary('label_high', self._high_res_cropped)
#         tf.image_summary('residual_reference', self._residual_cropped)
#         tf.image_summary('residual_inference', self._residual_inference)
#         tf.image_summary('inference', self._inference_image)


#     @property
#     def inference(self):
#         return self._inference_image

#     @property
#     def residual_inference(self):
#         return self._residual_inference

#     @property
#     def residual(self):
#         return self._residual_full

#     @property
#     def loss(self):
#         return self._loss

#     @property
#     def low_tensor(self):
#         return self._low_res_images

#     @property
#     def high_tensor(self):
#         return self._high_res_cropped

#     @property
#     def train_step(self):
#         return self._train_step


#     def _net_definition_2016(self):
#         k1 = 9
#         k2 = 5
#         n_channel_1 = 64
#         n_channel_2 = 32
#         conv1 = layer.conv_layer(self._low_res_images, [k1, k1, 1, n_channel_1], name="conv1")
#         conv2 = layer.conv_layer(conv1, [1, 1, n_channel_1, n_channel_2], name="full_connect")
#         residual_inference = layer.output_layer(conv2,
#                                                 [k2, k2, n_channel_2, 1],
#                                                 padding="VALID",
#                                                 name="output")
#         return residual_inference

#     def _net_definition_deep_conv(self):
#         n_channel = 128
#         kernel_shape = [3, 3, n_channel, n_channel]
#         conv1 = layer.conv_layer(self._low_res_images, [5, 5, 1, n_channel], name="conv_feature_1")
#         conv2 = layer.conv_layer(conv1, kernel_shape, name="conv_feature_2")
#         conv3 = layer.conv_layer(conv2, kernel_shape, name="conv_feature_3")
#         conv4 = layer.conv_layer(conv3, kernel_shape, name="conv_feature_4")
#         conv5 = layer.conv_layer(conv4, kernel_shape, name="conv_feature_5")
#         conv6 = layer.conv_layer(conv5, kernel_shape, name="conv_feature_6")
#         # conv7 = layer.conv_layer(conv6, kernel_shape, name="conv_feature_7")
#         # conv8 = layer.conv_layer(conv7, kernel_shape, name="conv_feature_8")
#         conv9 = layer.conv_layer(conv6, kernel_shape, name="full_connect")
#         conv10 = layer.conv_layer(conv9, kernel_shape, name="conv_recon_9")
#         conv11 = layer.conv_layer(conv10, kernel_shape, name="conv_recon_10")
#         conv12 = layer.conv_layer(conv11, kernel_shape, name="conv_recon_11")
#         conv13 = layer.conv_layer(conv12, kernel_shape, name="conv_recon_12")


#         residual_inference = layer.output_layer(conv13,
#                                                 [1, 1, n_channel, 1],
#                                                 name="output")
#         return residual_inference

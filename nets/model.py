"""
net definition
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import supernet.layers as layer

import supernet_old.supernet_input
FLAGS = tf.app.flags.FLAGS

class SuperNet(object):
    """Super resolution net
    """
    def __init__(self):
        self._global_step = tf.Variable(0, trainable=False, name='global_step')

        self._low_res_images = layer.input_layer([FLAGS.batch_size,
                                                  FLAGS.height,
                                                  FLAGS.width,
                                                  1],
                                                 "input_low_res")

        self._residual_full = layer.label_layer([FLAGS.batch_size,
                                                 FLAGS.height,
                                                 FLAGS.width,
                                                 1],
                                                "residual_reference")

        self._residual_cropped = layer.crop_layer(self._residual_full,
                                                  [FLAGS.valid_h,
                                                   FLAGS.valid_w],
                                                  [FLAGS.valid_y,
                                                   FLAGS.valid_x],
                                                  name='crop_residual')



        self._low_res_cropped = layer.crop_layer(self._low_res_images,
                                                 [FLAGS.valid_h,
                                                  FLAGS.valid_w],
                                                 [FLAGS.valid_y,
                                                  FLAGS.valid_x],
                                                 'crop_low_resolution')

        self._high_res_cropped = tf.add(self._low_res_cropped,
                                        self._residual_cropped,
                                        "label_high_res")

        self._residual_inference = self._net_definition_deep_conv()

        # self._residual_inference = self._net_definition_2016()

        self._loss = layer.l2_loss_layer(self._residual_cropped,
                                         self._residual_inference,
                                         name="loss")

        self._inference_image = tf.add(self._low_res_cropped,
                                       self._residual_inference,
                                       name="image_inference")

        # Decay the learning rate exponentially based on the number of steps.
        with tf.name_scope("train") as scope:
            learn_rate = tf.train.exponential_decay(FLAGS.learning_rate_init,
                                                    self._global_step,
                                                    FLAGS.decay_steps,
                                                    FLAGS.learning_rate_decay_factor,
                                                    staircase=True,
                                                    name=scope+'/learning_rate')

            tf.scalar_summary("learning rate", learn_rate)

            self._train_step = tf.train.AdamOptimizer(learn_rate).minimize(self._loss,
                                                                           self._global_step,
                                                                           name=scope+'/train_step')

            #self._train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss,global_step=step)

        tf.image_summary('input_low', self._low_res_cropped)
        tf.image_summary('label_high', self._high_res_cropped)
        tf.image_summary('residual_reference', self._residual_cropped)
        tf.image_summary('residual_inference', self._residual_inference)
        tf.image_summary('inference', self._inference_image)





    @property
    def inference(self):
        return self._inference_image

    @property
    def residual_inference(self):
        return self._residual_inference

    @property
    def residual(self):
        return self._residual_full

    @property
    def loss(self):
        return self._loss

    @property
    def low_tensor(self):
        return self._low_res_images

    @property
    def high_tensor(self):
        return self._high_res_cropped

    @property
    def train_step(self):
        return self._train_step


    def _net_definition_2016(self):
        k1 = 9
        k2 = 5
        n_channel_1 = 64
        n_channel_2 = 32
        conv1 = layer.conv_layer(self._low_res_images, [k1, k1, 1, n_channel_1], name="conv1")
        conv2 = layer.conv_layer(conv1, [1, 1, n_channel_1, n_channel_2], name="full_connect")
        residual_inference = layer.output_layer(conv2,
                                                [k2, k2, n_channel_2, 1],
                                                padding="VALID",
                                                name="output")
        return residual_inference

    def _net_definition_deep_conv(self):
        n_channel = 128
        kernel_shape = [3, 3, n_channel, n_channel]
        conv1 = layer.conv_layer(self._low_res_images, [5, 5, 1, n_channel], name="conv_feature_1")
        conv2 = layer.conv_layer(conv1, kernel_shape, name="conv_feature_2")
        conv3 = layer.conv_layer(conv2, kernel_shape, name="conv_feature_3")
        conv4 = layer.conv_layer(conv3, kernel_shape, name="conv_feature_4")
        conv5 = layer.conv_layer(conv4, kernel_shape, name="conv_feature_5")
        conv6 = layer.conv_layer(conv5, kernel_shape, name="conv_feature_6")
        # conv7 = layer.conv_layer(conv6, kernel_shape, name="conv_feature_7")
        # conv8 = layer.conv_layer(conv7, kernel_shape, name="conv_feature_8")
        conv9 = layer.conv_layer(conv6, kernel_shape, name="full_connect")
        conv10 = layer.conv_layer(conv9, kernel_shape, name="conv_recon_9")
        conv11 = layer.conv_layer(conv10, kernel_shape, name="conv_recon_10")
        conv12 = layer.conv_layer(conv11, kernel_shape, name="conv_recon_11")
        conv13 = layer.conv_layer(conv12, kernel_shape, name="conv_recon_12")


        residual_inference = layer.output_layer(conv13,
                                                [1, 1, n_channel, 1],
                                                name="output")
        return residual_inference


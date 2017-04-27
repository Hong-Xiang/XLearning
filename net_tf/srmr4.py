""" Net of SRSino8v3 """

import tensorflow as tf
import numpy as np
from ..utils.general import with_config
from ..models.image import inception_residual_block, align_by_crop


class SRSino8v4:
    @with_config
    def __init__(self,
                 input_shape,
                 batch_size,
                 filters=(64, 64, 64),
                 towers=4,
                 stem_filters=16,
                 is_bn=True,
                 depths=8,
                 crop_size=0,
                 log_dir='./log/',
                 model_dir='save',
                 learning_rate=1e-4,
                 is_adam=True,
                 is_clip=False,
                 **kwargs):
        self.filters = filters
        self.depths = depths
        self.input_shape = [None] + list(input_shape)
        self.crop_size = crop_size
        self.stages = len(filters)
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.is_bn = is_bn
        self.batch_size = batch_size
        self.learning_rate_value = learning_rate
        self.saver = None
        self.is_adam = is_adam
        self.is_clip = is_clip
        self.towers = towers
        self.stem_filters = stem_filters

    def predict_fullsize(self, ips, period):
        _, _, infer = self.predict(ips)
        batch_size = ips.shape[0]
        full_shape = list(self.input_shape)
        full_shape[0] = self.batch_size
        full_shape[2] *= 2
        infer_full = np.zeros(full_shape)
        infer_full[:, self.crop_size:-self.crop_size,
                   self.crop_size * 2:-self.crop_size * 2, :] = infer
        for i in range(2 * self.crop_size + 3):
            infer_full[:, :, i, :] = infer_full[:, :, i + period, :]
            infer_full[:, :, -i, :] = infer_full[:, :, -i - period, :]
        return infer_full

    def predict(self, ips):
        feed_dict = {self.ip: ips, self.training: False}
        return self.sess.run(self.infer, feed_dict=feed_dict)

    def save(self, ):
        self.saver.save(self.sess, self.model_dir,
                        global_step=self.global_step)

    def load(self, load_step=None):
        save_path = r'./' + self.model_dir + '-' + str(load_step)
        self.saver.restore(self.sess, save_path)
        self.global_step.assign(load_step)

    def train(self, ss):
        feed_dict = {self.ip: ss[0], self.lf: ss[1][2], self.training: True,
                     self.learning_rate: self.learning_rate_value}
        return self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)

    def summary(self, ss, is_train):
        feed_dict = {self.ip: ss[0], self.lf: ss[1][2], self.training: False,
                     self.learning_rate: self.learning_rate_value}
        sve = self.sess.run(self.summ_op, feed_dict=feed_dict)
        step = self.sess.run(self.global_step)
        if is_train:
            self.sw.add_summary(sve, global_step=step)
        else:
            self.sw_test.add_summary(sve, global_step=step)

    def build(self):
        self.learning_rate = tf.Variable(
            self.learning_rate_value, trainable=False, name='learning_rate')
        tf.summary.scalar('learning_rate', self.learning_rate)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.training = tf.placeholder(tf.bool, name='bn_switch')
        with tf.name_scope('inputs'):
            self.ip = tf.placeholder(
                tf.float32, self.input_shape, name='input')
            full_shape = list(self.input_shape)
            full_shape[2] *= 2
            self.lf = tf.placeholder(tf.float32, full_shape, name='label')

        with tf.name_scope('stem') as scope:
            h7 = tf.layers.conv2d(
                self.ip, filters=self.stem_filters, kernel_size=7, padding='same')
            h5 = tf.layers.conv2d(
                self.ip, filters=self.stem_filters, kernel_size=5, padding='same')
            h3 = tf.layers.conv2d(
                self.ip, filters=self.stem_filters, kernel_size=3, padding='same')
            h1 = tf.layers.conv2d(
                self.ip, filters=self.stem_filters, kernel_size=1, padding='same')
            h = tf.concat([h7, h5, h3, h1], axis=-1)
            h = tf.nn.crelu(h)

        with tf.name_scope('upscaling'):
            ipu = tf.image.resize_images(self.ip, full_shape[1:3])
            tf.summary.image('interp', ipu)

        with tf.name_scope('res_ref'):
            res_ref = self.lf - ipu
            tf.summary.image('img', res_ref)

        infers = []
        for i_stage, filters in enumerate(self.filters):
            with tf.name_scope('stage_%d' % i_stage):
                stage_in = h
                hs = [h] * self.towers
                for i_tower in range(self.towers):
                    with tf.name_scope('tower_%d' % i_tower):
                        for i_block in range(self.depths):
                            hs[i_tower] = inception_residual_block(hs[i_tower],
                                                                   filters // self.towers,
                                                                   is_final_activ=True,
                                                                   is_bn=self.is_bn,
                                                                   training=self.training,
                                                                   name='block_%d' % i_block)
                with tf.name_scope('infer'):
                    h = tf.concat(hs, axis=-1)
                    infer = tf.layers.conv2d_transpose(
                        h, 1, kernel_size=(1, 2), strides=(1, 2), padding='same')
                    infers.append(infer)

        with tf.name_scope('residual'):
            residual = tf.add_n(infers)
            tf.summary.image('img', residual)

        with tf.name_scope('infer'):
            self.infer = ipu + residual
            tf.summary.image('img', self.infer)

        with tf.name_scope('res_inf'):
            res_inf = self.lf - self.infer
            tf.summary.image('img', res_inf)

        with tf.name_scope('loss'):
            self.loss = tf.losses.mean_squared_error(self.lf, self.infer)/self.batch_size
            tf.summary.scalar('loss', self.loss)

        self.saver = tf.train.Saver()
        with tf.name_scope('optimizer'):
            if self.is_adam:
                opt = tf.train.AdamOptimizer(self.learning_rate)
            else:
                opt = tf.train.RMSPropOptimizer(self.learning_rate)
            if self.is_clip:
                gvs = opt.compute_gradients(self.loss)
                capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var)
                              for grad, var in gvs]
                self.train_op = opt.apply_gradients(
                    capped_gvs, global_step=self.global_step)
            else:
                self.train_op = opt.minimize(
                    self.loss, global_step=self.global_step)
        self.sess = tf.Session()
        self.summ_op = tf.summary.merge_all()
        self.sw = tf.summary.FileWriter(
            self.log_dir + 'train', self.sess.graph)
        self.sw_test = tf.summary.FileWriter(
            self.log_dir + 'test', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        print('SRMRv4 NET CONSTRUCTED.')

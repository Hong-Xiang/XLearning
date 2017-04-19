""" Net of SRSino8v3 """

import tensorflow as tf
import numpy as np
from ..utils.general import with_config
from ..models.image import inception_residual_block, align_by_crop


class SRSino8v3:
    @with_config
    def __init__(self,
                 input_shape,
                 batch_size,
                 filters=(64, 64, 64),
                 is_bn=True,
                 depths=8,
                 crop_size=0,
                 log_dir='./log/',
                 model_dir='save',
                 learning_rate=1e-4,
                 is_adam=True,
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
        self.learning_rate = learning_rate
        self.saver = None
        self.is_adam = is_adam

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
        return self.sess.run([self.inf_l, self.inf_r, self.infer], feed_dict=feed_dict)

    def save(self, ):
        self.saver.save(self.sess, self.model_dir,
                        global_step=self.global_step)

    def load(self, load_step=None):
        save_path = r'./' + self.model_dir + '-' + str(load_step)
        self.saver.restore(self.sess, save_path)
        self.global_step.assign(load_step)

    def train(self, ss):
        feed_dict = {self.ip: ss[0], self.ll: ss[1][0],
                     self.lr: ss[1][1], self.lf: ss[1][2], self.training: True}
        return self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)

    def summary(self, ss, is_train):
        feed_dict = {self.ip: ss[0], self.ll: ss[1][0],
                     self.lr: ss[1][1], self.lf: ss[1][2], self.training: False}
        sve = self.sess.run(self.summ_op, feed_dict=feed_dict)
        step = self.sess.run(self.global_step)
        if is_train:
            self.sw.add_summary(sve, global_step=step)
        else:
            self.sw_test.add_summary(sve, global_step=step)

    def build_infer(self, reps, input0, label, name=None):
        with tf.name_scope(name):
            with tf.name_scope('nin'):
                h = tf.layers.conv2d(reps, 512, 1, padding='same')
                h = tf.nn.crelu(h)
            with tf.name_scope('infer'):
                inf = tf.layers.conv2d(h, 1, 5, padding='same') + input0
            with tf.name_scope('crop'):
                input_shape = input0.shape.as_list()
                output_shape = [self.batch_size, input_shape[1] - 2 * self.crop_size,
                                input_shape[2] - 2 * self.crop_size, input_shape[3]]
                inf_c = tf.slice(
                    inf, [0, self.crop_size, self.crop_size, 0], output_shape)
                tf.summary.image('infer', inf_c)
            with tf.name_scope('residual'):
                with tf.name_scope('reference'):
                    ip_c, label_c = align_by_crop(
                        inf_c, [input0, label], batch_size=self.batch_size)
                    res_ip = label_c - ip_c
                    tf.summary.image('img', res_ip)
                with tf.name_scope('inference'):
                    res_infer = label_c - inf_c
                    loss = tf.losses.mean_squared_error(label_c, inf_c)
                    tf.summary.image('img', res_infer)
            tf.summary.image('infer', inf_c)
            tf.summary.image('input', ip_c)
            tf.summary.scalar('loss', loss)
            return inf_c, loss

    def build_infer_full(self, inf_l, inf_r, input0, label):
        with tf.name_scope('infer_full'):
            with tf.name_scope('inference'):
                inf_lr = tf.concat([inf_l, inf_r], axis=-1, name='inf_lr')
                infer_shape = list(inf_lr.shape.as_list())
                infer_shape[2] *= 2
                inf_up = tf.image.resize_images(inf_lr, infer_shape[1:3])
                h = tf.layers.conv2d(inf_up, 8, 1, padding='same')
                h = tf.nn.crelu(h)
                infer = tf.layers.conv2d(h, 1, 1, padding='same')
            tf.summary.image('infer', infer)
            with tf.name_scope('residual'):
                with tf.name_scope('reference'):
                    ip_c = align_by_crop(
                        inf_l, [input0], batch_size=self.batch_size)[0]
                    ip_up = tf.image.resize_images(ip_c, infer_shape[1:3])
                    label_c = align_by_crop(
                        infer, [label], batch_size=self.batch_size)[0]
                    res_itp = label_c - ip_up
                    tf.summary.image('img', res_itp)
                with tf.name_scope('inference'):
                    res_infer = label_c - infer
                    loss = tf.losses.mean_squared_error(label_c, infer)
                    tf.summary.image('img', res_infer)
            tf.summary.scalar('loss', loss)
            tf.summary.image('label', label_c)
            return infer, loss

    def build(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.training = tf.placeholder(tf.bool, name='bn_switch')
        with tf.name_scope('inputs'):
            self.ip = tf.placeholder(
                tf.float32, self.input_shape, name='input')
            self.ll = tf.placeholder(
                tf.float32, self.input_shape, name='label_l')
            self.lr = tf.placeholder(
                tf.float32, self.input_shape, name='label_r')
            full_shape = list(self.input_shape)
            full_shape[2] *= 2
            self.lf = tf.placeholder(tf.float32, full_shape, name='label_full')

        with tf.name_scope('stem') as scope:
            h = tf.layers.conv2d(
                self.ip, filters=64, kernel_size=7, padding='same')
            h = tf.nn.crelu(h)

        reps = []
        reps.append(self.ip)
        for i_stage, f in enumerate(self.filters):
            with tf.name_scope('stage'):
                with tf.name_scope('nin'):
                    h = tf.layers.conv2d(h, f, 1, padding='same')
                    h = tf.layers.batch_normalization(
                        h, training=self.training)
                    h = tf.nn.crelu(h)
                for i_block in range(self.depths):
                    with tf.name_scope('block'):
                        h = inception_residual_block(
                            h, f, is_bn=self.is_bn, training=self.training)
            reps.append(h)

        with tf.name_scope('concat'):
            h = tf.concat(reps, axis=-1)

        self.inf_l, loss_l = self.build_infer(
            h, self.ip, self.ll, name='infer_l')
        self.inf_r, loss_r = self.build_infer(
            h, self.ip, self.lr, name='infer_r')

        self.infer, loss_f = self.build_infer_full(
            self.inf_l, self.inf_r, self.ip, self.lf)

        with tf.name_scope('loss_merge'):
            self.loss = loss_l + loss_r + loss_f
        tf.summary.scalar('loss', self.loss)

        self.saver = tf.train.Saver()
        with tf.name_scope('optimizer'):
            if self.is_adam:
                opt = tf.train.AdamOptimizer(self.learning_rate)
            else:
                opt = tf.train.RMSPropOptimizer(self.learning_rate)
            self.train_op = opt.minimize(
                self.loss, global_step=self.global_step)
        self.sess = tf.Session()
        self.summ_op = tf.summary.merge_all()
        self.sw = tf.summary.FileWriter(
            self.log_dir + 'train', self.sess.graph)
        self.sw_test = tf.summary.FileWriter(
            self.log_dir + 'test', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        print('NET CONSTRUCTED.')

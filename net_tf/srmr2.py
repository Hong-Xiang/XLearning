import tensorflow as tf
import numpy as np
from ..utils.general import with_config


class SRSino8v2:
    @with_config
    def __init__(self,
                 input_shape,
                 batch_size,
                 filters=64,
                 depths=2,
                 blocks=16,
                 cores=3,
                 is_bn=True,
                 crop_size=0,
                 log_dir='./log/',
                 model_dir='save',
                 lrs=1e-4,
                 **kwargs):
        self.filters = filters
        self.depths = depths
        self.blocks = blocks
        self.input_shape = [None] + list(input_shape)
        self.cropped_shape = [None, input_shape[0] - crop_size * 2,
                              input_shape[1] - crop_size * 2, 1]
        self.crop_size = crop_size
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.is_bn = is_bn
        self.lrs = lrs
        self.saver = None
        self.cores = cores
        self.batch_size = batch_size

    def predict_single(self, net_name, ss):
        if net_name == 'net_8x':
            ops = self.ops8x
        if net_name == 'net_4x':
            ops = self.ops4x
        if net_name == 'net_2x':
            ops = self.ops2x

        res_l = ops['res_l']
        res_r = ops['res_r']
        inf_l = ops['inf_l']
        inf_r = ops['inf_r']
        res_ll = ops['res_ll']
        res_lr = ops['res_lr']
        ip_c = self.ipc
        ll = self.ll
        lr = self.lr
        run_ops = [res_l, res_r, inf_l, inf_r, res_ll, res_lr, ip_c, ll, lr]
        ll_c = ss[1][0][:, self.crop_size:-self.crop_size,
                        self.crop_size:-self.crop_size, :]
        lr_c = ss[1][1][:, self.crop_size:-self.crop_size,
                        self.crop_size:-self.crop_size, :]
        feed_dict = {self.ip: ss[0], self.ll: ll_c,
                     self.lr: lr_c, self.training: False}
        results = self.sess.run(run_ops, feed_dict=feed_dict)
        errl = results[2] - ll_c
        errr = results[3] - lr_c
        results += [errl, errr]
        names = ['res_l', 'res_r', 'inf_l', 'inf_r', 'res_ll',
                 'res_lr', 'ipc', 'll', 'lr', 'ip', 'errl', 'errr']
        for n, arr in zip(names, results):
            fn = os.join(os.path.abspath('results'), n)
            np.save(fn, arr)

    def predict(self, ips):
        hight = ips.shape[1]
        width = ips.shape[2]
        if hight != 363 or width != 90:
            raise ValueError('Invalid input shape.')
        inf_l = self.sess.run(self.ops8x['inf_l'], feed_dict={
                              self.ip: ips, self.training: False})
        inf_r = self.sess.run(self.ops8x['inf_r'], feed_dict={
                              self.ip: ips, self.training: False})
        inf_l_pad = np.pad(
            inf_l, ((0, 0), [self.crop_size] * 2, [self.crop_size] * 2, (0, 0)))
        inf_r_pad = np.pad(
            inf_r, ((0, 0), [self.crop_size] * 2, [self.crop_size] * 2, (0, 0)))
        inf4x = np.zeros([1, 363, 180, 1])
        inf4x[0, :, ::2, :] = inf_l_pad
        inf4x[0, :, 1::2, :] = inf_r_pad
        inf4x[0, :, :self.crop_size, :] = inf4x[0, :, 180:180 + 64, ]
        pass

    def save(self, ):
        self.saver.save(self.sess, self.model_dir,
                        global_step=self.global_step)

    def load(self, load_step=None):
        save_path = self.model_dir + str(load_step)
        self.saver.restore(net.sess, save_path)
        self.global_step.assign(load_step)

    def train(self, net_name, ss):
        if net_name == 'net_8x':
            train_op = self.train_8x
            loss_op = self.loss8x
        if net_name == 'net_4x':
            train_op = self.train_4x
            loss_op = self.loss4x
        if net_name == 'net_2x':
            train_op = self.train_2x
            loss_op = self.loss2x
        ll_c = ss[1][0][:, self.crop_size:-self.crop_size,
                        self.crop_size:-self.crop_size, :]
        lr_c = ss[1][1][:, self.crop_size:-self.crop_size,
                        self.crop_size:-self.crop_size, :]
        feed_dict = {self.ip: ss[0], self.ll: ll_c,
                     self.lr: lr_c, self.training: True}
        return self.sess.run([loss_op, train_op], feed_dict=feed_dict)

    def summary(self, net_name, ss, is_train):
        if net_name == 'net_8x':
            summ_op = self.summ8x
        if net_name == 'net_4x':
            summ_op = self.summ4x
        if net_name == 'net_2x':
            summ_op = self.summ2x
        ll_c = ss[1][0][:, self.crop_size:-self.crop_size,
                        self.crop_size:-self.crop_size, :]
        lr_c = ss[1][1][:, self.crop_size:-self.crop_size,
                        self.crop_size:-self.crop_size, :]
        feed_dict = {self.ip: ss[0], self.ll: ll_c,
                     self.lr: lr_c, self.training: False}
        sve = self.sess.run(summ_op, feed_dict=feed_dict)
        step = self.sess.run(self.global_step)
        # for sve in sv:
        if is_train:
            self.sw.add_summary(sve, global_step=step)
        else:
            self.sw_test.add_summary(sve, global_step=step)

    def build_core(self, ip, conv_c, bn_c):
        h0 = ip
        h = ip
        for i in range(self.blocks):
            with tf.name_scope('block_%d' % i):
                hc = h
                for j in range(self.depths):
                    with tf.name_scope('conv_%d' % j) as scope:
                        hc = tf.layers.conv2d(hc, **conv_c)
                        if self.is_bn:
                            hc = tf.layers.batch_normalization(hc, **bn_c)
                        hc = tf.nn.crelu(hc)
                with tf.name_scope('conv_end') as scope:
                    hc = tf.layers.conv2d(hc, **conv_c)
                    if self.is_bn:
                        hc = tf.layers.batch_normalization(hc, **bn_c)
                with tf.name_scope('add'):
                    h = h + hc
        with tf.name_scope('concate'):
            h = tf.concat([h0, h], axis=-1)
        return h

    def build_infer(self, reps):
        summs = []
        with tf.name_scope('res'):
            with tf.name_scope('conv_l'):
                res_l = tf.layers.conv2d(reps, 1, 3, padding='same')
            with tf.name_scope('conv_r'):
                res_r = tf.layers.conv2d(reps, 1, 3, padding='same')
            with tf.name_scope('crop'):
                res_l = tf.slice(res_l,
                                 [0, self.crop_size, self.crop_size, 0],
                                 [self.batch_size, self.cropped_shape[1], self.cropped_shape[2], 1], name='ip_c')
                res_r = tf.slice(res_r,
                                 [0, self.crop_size, self.crop_size, 0],
                                 [self.batch_size, self.cropped_shape[1], self.cropped_shape[2], 1], name='ip_c')
            summs.append(tf.summary.image('res_l', res_l))
            summs.append(tf.summary.image('res_r', res_r))
        with tf.name_scope('infer'):
            inf_l = self.ipc + res_l
            inf_r = self.ipc + res_r
            summs.append(tf.summary.image('inf_l', inf_l))
            summs.append(tf.summary.image('inf_r', inf_r))
        with tf.name_scope('res_ip'):
            res_ll = self.ll - self.ipc
            res_lr = self.lr - self.ipc
            summs.append(tf.summary.image('res_ll', res_ll))
            summs.append(tf.summary.image('res_lr', res_lr))
        with tf.name_scope('loss'):
            loss_l = tf.losses.mean_squared_error(inf_l, self.ll)
            loss_r = tf.losses.mean_squared_error(inf_r, self.lr)
            loss = loss_l + loss_r
            summs.append(tf.summary.scalar('loss', loss))
        ops = {
            'res_l': res_l,
            'res_r': res_r,
            'inf_l': inf_l,
            'inf_r': inf_r,
            'res_ll': res_ll,
            'res_lr': res_lr
        }
        return ops, loss, summs

    def build(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.training = tf.placeholder(tf.bool, name='bn_switch')
        summ_ip = []
        with tf.name_scope('input'):
            self.ip = tf.placeholder(
                tf.float32, self.input_shape, name='input')
            self.ipc = tf.slice(self.ip,
                                [0, self.crop_size, self.crop_size, 0],
                                [self.batch_size, self.cropped_shape[1], self.cropped_shape[2], 1], name='ip_c')
            self.ll = tf.placeholder(
                tf.float32, self.cropped_shape, name='label_l')
            self.lr = tf.placeholder(
                tf.float32, self.cropped_shape, name='label_r')
            summ_ip.append(tf.summary.image('input_crop', self.ipc))
            summ_ip.append(tf.summary.image('label_l_crop', self.ll))
            summ_ip.append(tf.summary.image('label_r_crop', self.lr))

        conv_c = {
            'padding': 'same',
            'filters': self.filters,
            'kernel_size': 3
        }
        bn_c = {
            'training': self.training
        }
        with tf.name_scope('init_reps') as scope:
            h = tf.layers.conv2d(self.ip, name=scope + 'conv', **conv_c)
            if self.is_bn:
                h = tf.layers.batch_normalization(h, name=scope + 'bn', **bn_c)
        for i in range(self.cores):
            conv_c['filters'] = self.filters * (2**i)
            with tf.name_scope('core_%d' % i):
                h = self.build_core(h, conv_c, bn_c)
        with tf.name_scope('infer8x'):
            self.ops8x, self.loss8x, self.summ8x = self.build_infer(h)
        with tf.name_scope('infer4x'):
            self.ops4x, self.loss4x, self.summ4x = self.build_infer(h)
        with tf.name_scope('infer2x'):
            self.ops2x, self.loss2x, self.summ2x = self.build_infer(h)
        self.summ8x += summ_ip
        self.summ4x += summ_ip
        self.summ2x += summ_ip
        self.summ8x = tf.summary.merge(self.summ8x)
        self.summ4x = tf.summary.merge(self.summ4x)
        self.summ2x = tf.summary.merge(self.summ2x)
        self.saver = tf.train.Saver()
        with tf.name_scope('optimizer'):
            opt = tf.train.AdamOptimizer(1e-4)
            self.train_8x = opt.minimize(
                self.loss8x, global_step=self.global_step)
            self.train_4x = opt.minimize(
                self.loss4x, global_step=self.global_step)
            self.train_2x = opt.minimize(
                self.loss2x, global_step=self.global_step)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sw = tf.summary.FileWriter(
            self.log_dir + 'train', self.sess.graph)
        self.sw_test = tf.summary.FileWriter(
            self.log_dir + 'test', self.sess.graph)

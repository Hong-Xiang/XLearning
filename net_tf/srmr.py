import tensorflow as tf
from ..utils.general import with_config
import os


class SRSino8:
    @with_config
    def __init__(self,
                 input_shape,
                 filters_8x=4,
                 depths_8x=1,
                 blocks_8x=2,
                 filters_4x=4,
                 depths_4x=1,
                 blocks_4x=2,
                 filters_2x=4,
                 depths_2x=1,
                 blocks_2x=2,
                 is_bn=True,
                 is_crop=False,
                 log_dir='.',
                 model_dir='.',
                 lrs=1e-4,
                 **kwargs):
        self.filters_8x = filters_8x
        self.depths_8x = depths_8x
        self.blocks_8x = blocks_8x
        self.filters_4x = filters_4x
        self.depths_4x = depths_4x
        self.blocks_4x = blocks_4x
        self.filters_2x = filters_2x
        self.depths_2x = depths_2x
        self.blocks_2x = blocks_2x
        self.input_shape = [None] + list(input_shape)
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.is_bn = is_bn
        self.trainable = None
        self.saver8x = None
        self.saver4x = None
        self.svaer2x = None
        self.lrs = lrs

    def _conv_block(self,
                    ip,
                    filters,
                    scope,
                    is_bn=None,
                    is_active=True,
                    is_train=True,
                    reuse=None):
        with tf.name_scope('conv'):
            h = tf.layers.conv2d(ip, filters, 3, name=scope +
                                 'conv', reuse=reuse, trainable=is_train, padding='same')
            if is_bn is None:
                is_bn = self.is_bn
            if is_bn:
                h = tf.layers.batch_normalization(
                    h, reuse=reuse, training=is_train, scale=False, name=scope + 'bn', trainable=is_train)
            if is_active:
                h = tf.nn.elu(h, name='elu')
        return h

    def _build_core(self, ip, label, filters, depths, blocks, reuse=None, scope=None, is_train=True):
        cfg = {'reuse': reuse, 'is_train': is_train, 'filters': filters}
        h = self._conv_block(ip, scope=scope + 'init/', **cfg)
        if is_train:
            summary_prefix = scope + 'train/'
        else:
            summary_prefix = scope + 'test/'
        for i in range(blocks):
            with tf.name_scope('block_%d' % i) as s_block:
                hc = h
                for j in range(depths):
                    with tf.name_scope('conv_%d' % j) as s_block_c:
                        hc = self._conv_block(
                            hc, scope=scope + 'block_%d/' % i + 'conv_%d/' % j, **cfg)
                with tf.name_scope('conv_end') as s_block_ce:
                    hc = self._conv_block(
                        hc, scope=scope + 'block_%d/' % i + 'conv_end', is_active=False, **cfg)
                with tf.name_scope('add'):
                    h = h + hc
        with tf.name_scope('infer_l') as s_inf_l:
            outl = self._conv_block(
                h, filters=1, scope=scope + 'infer_l/', is_active=False, is_train=is_train, reuse=reuse)
            tf.summary.image(summary_prefix + 'res_l_inf', outl)
            with tf.name_scope('add_ip'):
                outl = outl + ip
        with tf.name_scope('infer_r') as s_inf_r:
            outr = self._conv_block(
                h, filters=1, scope=scope + 'infer_r/', is_active=False, is_train=is_train, reuse=reuse)
            tf.summary.image(summary_prefix + 'res_r_inf', outr)
            with tf.name_scope('add_ip'):
                outr = outr + ip
        with tf.name_scope('loss'):
            with tf.name_scope('loss_l'):
                lossl = tf.losses.mean_squared_error(label[0], outl)
            with tf.name_scope('loss_r'):
                lossr = tf.losses.mean_squared_error(label[1], outr)
            with tf.name_scope('loss'):
                loss = lossl + lossr
            with tf.name_scope('residual'):
                res_l = label[0] - ip
                res_r = label[1] - ip
            tf.summary.image('res_l_ref', res_l)
            tf.summary.image('res_r_ref', res_r)
            tf.summary.scalar(summary_prefix + 'loss', loss)
        return outl, outr, loss

    def _build_8x(self):
        filters = self.filters_8x
        depths = self.depths_8x
        blocks = self.blocks_8x
        ptr = self.phs['net_8x']['train']
        ptt = self.phs['net_8x']['test']
        with tf.name_scope('net_8x'):
            with tf.name_scope('graph') as scope:
                outl, outr, loss = self._build_core(
                    ptr['ip'], (ptr['ll'], ptr['lr']), filters=filters, depths=depths, blocks=blocks, reuse=False, scope=scope, is_train=True)
            with tf.name_scope('train'):
                outl, outr, loss = self._build_core(
                    ptr['ip'], (ptr['ll'], ptr['lr']), filters=filters, depths=depths, blocks=blocks, reuse=True, scope=scope, is_train=True)
            with tf.name_scope('predict'):
                outl_test, outr_test, loss_test = self._build_core(
                    ptt['ip'], (ptt['ll'], ptt['lr']), filters=filters, depths=depths, blocks=blocks, reuse=True, scope=scope, is_train=False)
        self.outl8x = outl
        self.outr8x = outr
        self.loss8x = loss
        self.outl8x_test = outl_test
        self.outr8x_test = outr_test
        self.loss8x_test = loss_test

    def _build_4x(self):
        filters = self.filters_4x
        depths = self.depths_4x
        blocks = self.blocks_4x
        ptr = self.phs['net_4x']['train']
        ptt = self.phs['net_4x']['test']
        with tf.name_scope('net_4x') as scope:
            with tf.name_scope('graph') as scope:
                outl, outr, loss = self._build_core(
                    ptr['ip'], (ptr['ll'], ptr['lr']), filters=filters, depths=depths, blocks=blocks, reuse=False, scope=scope, is_train=True)
            with tf.name_scope('train'):
                outl, outr, loss = self._build_core(
                    ptr['ip'], (ptr['ll'], ptr['lr']), filters=filters, depths=depths, blocks=blocks, reuse=True, scope=scope, is_train=True)
            with tf.name_scope('predict'):
                outl_test, outr_test, loss_test = self._build_core(
                    ptt['ip'], (ptt['ll'], ptt['lr']), filters=filters, depths=depths, blocks=blocks, reuse=True, scope=scope, is_train=False)
        self.outl4x = outl
        self.outr4x = outr
        self.loss4x = loss
        self.outl4x_test = outl_test
        self.outr4x_test = outr_test
        self.loss4x_test = loss_test

    def _build_2x(self):
        filters = self.filters_2x
        depths = self.depths_2x
        blocks = self.blocks_2x
        ptr = self.phs['net_2x']['train']
        ptt = self.phs['net_2x']['test']
        with tf.name_scope('net_2x') as scope:
            with tf.name_scope('graph') as scope:
                outl, outr, loss = self._build_core(
                    ptr['ip'], (ptr['ll'], ptr['lr']), filters=filters, depths=depths, blocks=blocks, reuse=False, scope=scope, is_train=True)
            with tf.name_scope('train'):
                outl, outr, loss = self._build_core(
                    ptr['ip'], (ptr['ll'], ptr['lr']), filters=filters, depths=depths, blocks=blocks, reuse=True, scope=scope, is_train=True)
            with tf.name_scope('predict'):
                outl_test, outr_test, loss_test = self._build_core(
                    ptt['ip'], (ptt['ll'], ptt['lr']), filters=filters, depths=depths, blocks=blocks, reuse=True, scope=scope, is_train=False)
        self.outl2x = outl
        self.outr2x = outr
        self.loss2x = loss
        self.outl2x_test = outl_test
        self.outr2x_test = outr_test
        self.loss2x_test = loss_test

    def predict(self, input):
        pass

    def save(self, net_name=None):
        if net_name is None:
            self.save('net_8x')
            self.save('net_4x')
            self.save('net_2x')
            return None
        if is_save8x:
            self.saver8x.save(self.sess, os.path.join(
                self.model_dir, 'net_8x'), global_step=self.global_step)
        if net_name == 'net_4x':
            self.saver4x.save(self.sess, os.path.join(
                self.model_dir, 'net_4x'), global_step=self.global_step)
        if net_name == 'net_2x':
            self.saver2x.save(self.sess, os.path.join(
                self.model_dir, 'net_2x'), global_step=self.global_step)

    def load(self, model_name=None, load_step=None):
        if model_name is None:
            self.load('net_8x', load_step)
            self.load('net_4x', load_step)
            self.load('net_2x', load_step)
            return None
        if model_name == 'net_8x':
            save_path = os.path.join(self.model_dir, 'net_8x')
            self.saver8x.restore(net.sess, save_path)
        if model_name == 'net_4x':
            save_path = os.path.join(self.model_dir, 'net_4x')
            self.saver4x.restore(net.sess, save_path)
        if model_name == 'net_2x':
            save_path = os.path.join(self.model_dir, 'net_2x')
            self.saver2x.restore(net.sess, save_path)
        self.global_step.assign(load_step)

    def __def_phs(self, net_name):
        phs = {}
        phs['train'] = {}
        phs['test'] = {}
        with tf.name_scope(net_name):
            with tf.name_scope('train'):
                phs['train']['ip'] = tf.placeholder(
                    tf.float32, self.input_shape, 'ip')
                phs['train']['ll'] = tf.placeholder(
                    tf.float32, self.input_shape, 'll')
                phs['train']['lr'] = tf.placeholder(
                    tf.float32, self.input_shape, 'lr')
            with tf.name_scope('test'):
                phs['test']['ip'] = tf.placeholder(
                    tf.float32, self.input_shape, 'ip')
                phs['test']['ll'] = tf.placeholder(
                    tf.float32, self.input_shape, 'll')
                phs['test']['lr'] = tf.placeholder(
                    tf.float32, self.input_shape, 'lr')
        return phs
    
    def train(self, net_name, ss):
        if net_name == 'net_8x':
            train_op = self.train_8x
            feed_dict = {
                self.phs['net_8x']['train']['ip']: ss[0],
                self.phs['net_8x']['train']['il']: ss[1][0],
                self.phs['net_8x']['train']['ir']: ss[1][1]
            }
        if net_name == 'net_4x':
            train_op = self.train_4x
            feed_dict = {
                self.phs['net_4x']['train']['ip']: ss[0],
                self.phs['net_4x']['train']['il']: ss[1][0],
                self.phs['net_4x']['train']['ir']: ss[1][1]
            }
        if net_name == 'net_2x':
            train_op = self.train_2x
            feed_dict = {
                self.phs['net_2x']['train']['ip']: ss[0],
                self.phs['net_2x']['train']['il']: ss[1][0],
                self.phs['net_2x']['train']['ir']: ss[1][1]
            }
        self.sess.run(train_op, feed_dict=feed_dict)
    
    def summary(self, sample_dict):
        ss8x = sample_dict['ss8x']
        ss4x = sample_dict['ss4x']
        ss2x = sample_dict['ss2x']


    def build(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        with tf.name_scope('inputs'):
            self.phs = {}
            self.phs['net_8x'] = self.__def_phs('net_8x')
            self.phs['net_4x'] = self.__def_phs('net_4x')
            self.phs['net_2x'] = self.__def_phs('net_2x')

        opt = tf.train.AdamOptimizer(1e-4)
        self._build_8x()
        print('build_8x done.')
        self.train_8x = opt.minimize(self.loss8x, global_step=self.global_step)
        print('train_8x done.')
        self.var_8x = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='net_8x')
        self._build_4x()
        print('build_4x done.')
        self.train_4x = opt.minimize(self.loss4x, global_step=self.global_step)
        print('train_4x done.')
        self.var_4x = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='net_4x')
        self._build_2x()
        print('build_2x done.')
        self.train_2x = opt.minimize(self.loss2x, global_step=self.global_step)
        print('train_2x done.')
        self.var_2x = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='net_2x')
        self.saver8x = tf.train.Saver(var_list=self.var_8x)
        self.saver4x = tf.train.Saver(var_list=self.var_4x)
        self.saver2x = tf.train.Saver(var_list=self.var_2x)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.summary_op = tf.summary.merge_all()
        # print('\n'.join([v.name for v in self.var_8x]))
        self.sw = tf.summary.FileWriter(self.log_dir, self.sess.graph)

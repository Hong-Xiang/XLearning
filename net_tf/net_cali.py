import tensorflow as tf
from ..utils.general import with_config


class CaliNet:
    @with_config
    def __init__(self,
                 lr):
        self.lr = lr

    def build(self):
        self.ip = tf.placeholder(tf.float32, [None, 10, 10, 1], name='input')
        tf.summary.image('input', self.ip)
        self.label = tf.placeholder(tf.float32, [None, 1])
        with tf.name_scope('convs'):
            h = tf.layers.conv2d(self.ip, 64, 3)
            h = tf.crelu(h)
            h = tf.layers.conv2d(h, 128, 3)
            h = tf.crelu(h)
            h = tf.layers.conv2d(h, 256, 3)
            h = tf.crelu(h)
            h = tf.layers.conv2d(h, 512, 3)
            h = tf.crelu(h)
            repc = tf.reshape(h, [None, -1])
        with tf.name_scope('denses'):
            h = tf.reshape(self.ip, [None, -1])
            h = tf.layers.dense(h, 128)
            h = tf.nn.elu(h)
            h = tf.layers.dense(h, 256)
            h = tf.nn.elu(h)
            h = tf.layers.dense(h, 512)
            h = tf.nn.elu(h)
            h = tf.layers.dense(h, 1024)
            repd = tf.nn.elu(h)
        with tf.name_scope('regre'):
            h = tf.concat([repc, repd], axis=-1)
            self.out = tf.layers.dense(h, 1)
        with tf.name_scope('loss'):
            self.loss = tf.losses.mean_squared_error(self.out, self.label)
            tf.summary.scalar('loss', self.loss)
        with tf.name_scope('train'):
            opt = tf.train.AdamOptimizer(self.lr)
            self.train = opt.minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sw = tf.summary.FileWriter('./log', self.sess.graph)




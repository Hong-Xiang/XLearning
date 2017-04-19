import tensorflow as tf
from ..utils.general import with_config


class CaliNet:
    @with_config
    def __init__(self,
                 lr,
                 **kwargs):
        self.lr = lr

    def build(self):
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.ip = tf.placeholder(tf.float32, [None, 10, 10, 1], name='input')
        tf.summary.image('input', self.ip)
        self.label = tf.placeholder(tf.float32, [None, 1])
        with tf.name_scope('convs'):
            h = tf.layers.conv2d(self.ip, 64, 3)
            h = tf.nn.crelu(h)
            h = tf.layers.conv2d(h, 128, 3)
            h = tf.nn.crelu(h)
            h = tf.layers.conv2d(h, 256, 3)
            h = tf.nn.crelu(h)
            h = tf.layers.conv2d(h, 512, 3)
            h = tf.nn.crelu(h)
            repc = tf.contrib.layers.flatten(h)
        with tf.name_scope('denses'):
            h = tf.contrib.layers.flatten(self.ip)
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
            self.train = opt.minimize(self.loss, global_step=self.global_step)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sw = tf.summary.FileWriter('./log', self.sess.graph)
        self.saver = tf.train.Saver()
        self.sm = tf.summary.merge_all()
    
    def train(self, img, lab):
        _, loss_v = self.sess.run([self.train, self.loss], feed_dict={self.ip: img, self.label: lab})
        return loss_v

        
        


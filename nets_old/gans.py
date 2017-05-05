import tensorflow as tf
import numpy as np
from keras.layers import Dense, Convolution2D, BatchNormalization, ELU, Flatten, Deconvolution2D, Reshape, UpSampling2D, Dropout
from keras.objectives import mean_squared_error
from keras.models import Sequential
from .base import Net, NetGen
from ..utils.general import with_config
from ..model.layers import Input, Label, Denses
from ..keras_ext.constrains import MaxMinValue


class LSGAN(NetGen):

    @with_config
    def __init__(self,
                 settings=None,
                 gen_freq=5,
                 pre_train=100,
                 **kwargs):
        NetGen.__init__(self, **kwargs)
        self._settings = settings
        self._models_names = ['WGan', 'Cri', 'Gen']
        self._is_train_step = [True, True, False]
        self.gen_freq = self._update_settings('gen_freq', gen_freq)
        self.pre_train = self._update_settings('pre_train', pre_train)
        a = -1
        b = 1
        c = 0
        self._la = tf.Variable(np.ones(shape=(
            self._batch_size, 1)) * a, trainable=False, name='a_label', dtype=tf.float32)
        self._lb = tf.Variable(np.ones(shape=(
            self._batch_size, 1)) * b, trainable=False, name='b_label', dtype=tf.float32)
        self._lc = tf.Variable(np.ones(shape=(
            self._batch_size, 1)) * c, trainable=False, name='c_label', dtype=tf.float32)

    def _define_losses(self):
        with tf.name_scope('loss_gen'):
            loss_gen = tf.losses.mean_squared_error(self._lc, self._logit_fake)
        with tf.name_scope('loss_cri'):
            loss_cri = tf.losses.mean_squared_error(
                self._la, self._logit_fake) + tf.losses.mean_squared_error(self._lb, self._logit_true)

        self._losses[0] = loss_gen
        self._losses[1] = loss_cri
        self._losses[2] = None

        tf.summary.scalar('loss_gen', loss_gen)
        tf.summary.scalar('loss_cri', loss_cri)

    def _define_optims(self):
        with tf.name_scope('optim_gen'):
            optim_gen = tf.train.RMSPropOptimizer(self._lrs_tensor[0])

        with tf.name_scope('optim_cri'):
            optim_cri = tf.train.RMSPropOptimizer(self._lrs_tensor[0])

        self._optims[0] = optim_gen
        self._optims[1] = optim_cri

    def _define_train_steps(self):
        with tf.name_scope('train_cri'):
            cri_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='cri')
            self._train_steps[1] = self._optims[1].minimize(
                self._losses[1], var_list=cri_vars)
        with tf.name_scope('train_gen'):
            gen_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
            self._train_steps[0] = self._optims[0].minimize(
                self._losses[0], var_list=gen_vars)

    def _define_models_dense(self):
        x = self._latent_input
        with tf.name_scope('gen'):
            with tf.name_scope("denses"):
                x = Denses((self._latent_dims,), 28*28, self._hiddens)(x)
            with tf.name_scope('reshape'):
                self._generated = tf.reshape(
                    x, shape=(self._batch_size, 28, 28, 1))
        tf.summary.image('generated', self._generated)
        with tf.name_scope('cri'):
            with tf.name_scope("flatten"):
                f_data = tf.reshape(self._data, (self._batch_size, 28 * 28))
                f_fake = tf.reshape(
                    self._generated, (self._batch_size, 28 * 28))
            with tf.name_scope('denses'):
                cri = Denses((28*28,), 1, self._hiddens)
            self._logit_fake = cri(f_fake)
            self._logit_true = cri(f_data)

    def _define_models_convolution(self):
        # build generator

        x = self._latent_input
        with tf.name_scope('gen'):
            with tf.name_scope('denses'):
                x = Dense(1024, activation='elu')(x)
                x = Dense(128 * 7 * 7, activation='elu')(x)
            with tf.name_scope('reshape'):
                x = tf.reshape(x, shape=(self._batch_size, 7, 7, 128))
            with tf.name_scope('upsampling0'):
                x = UpSampling2D(size=(2, 2))(x)
                x = Convolution2D(128, 5, 5, border_mode='same',
                                  activation='elu', init='glorot_normal')(x)
                x = Convolution2D(256, 5, 5, border_mode='same',
                                  activation='elu', init='glorot_normal')(x)
            with tf.name_scope('upsampling1'):
                x = UpSampling2D(size=(2, 2))(x)
                x = Convolution2D(256, 5, 5, border_mode='same',
                                  activation='elu', init='glorot_normal')(x)
                x = Convolution2D(512, 5, 5, border_mode='same',
                                  activation='elu', init='glorot_normal')(x)
            with tf.name_scope('generation'):
                x = Convolution2D(1, 2, 2, border_mode='same',
                                  init='glorot_normal')(x)
                self._generated = x
                tf.summary.image('generated', self._generated, max_outputs=16)

        # build discriminator

        self._cri_weight_names = []
        with tf.name_scope('cri'):
            cri = Sequential()
            with tf.name_scope('conv_0'):
                c = Convolution2D(32, 3, 3, subsample=(
                    2, 2), activation='elu', input_shape=(28, 28, 1))
                cri.add(c)
                cri.add(Dropout(0.3))
            with tf.name_scope('conv_1'):
                c = Convolution2D(64, 3, 3, activation='elu')
                cri.add(c)
                cri.add(Dropout(0.3))
            with tf.name_scope('conv_2'):
                c = Convolution2D(
                    128, 3, 3, subsample=(2, 2), activation='elu')
                cri.add(c)
                cri.add(Dropout(0.3))
            with tf.name_scope('conv_3'):
                c = Convolution2D(256, 3, 3, activation='elu')
                cri.add(c)
                cri.add(Dropout(0.3))
            with tf.name_scope('flatten'):
                cri.add(Flatten())
            with tf.name_scope('logits'):
                d = Dense(1)
                cri.add(d)
            self._logit_fake = cri(self._generated)
            self._logit_true = cri(self._data)

    def _define_models(self):
        with tf.name_scope('input'):
            self._data = Input(shape=self._inputs_dims[0], name='data_input')
        with tf.name_scope('latent_input'):
            self._latent_input = Input(shape=(self._latent_dims,))

        if self._arch == 'default' or self._arch == 'dense':
            self._define_models_dense()
        else:
            self._define_models_convolution()

        self._inputs[0] = [self._data, self._latent_input]
        self._outputs[0] = [self._logit_fake]

        self._inputs[1] = [self._data, self._latent_input]
        self._outputs[1] = [self._logit_true, self._logit_fake]

        self._inputs[2] = [self._latent_input]
        self._outputs[2] = [self._generated]

""" Implementation of AAE network on MNIST """

import numpy as np
import tensorflow as tf
from keras.layers import Input, Convolution2D, Dense
from .base import Net
from ..utils.general import with_config


class AAE1D(Net):

    @with_config
    def __init__(self,
                 latent_dims=2,
                 lsgan_a=-1.0,
                 lsgan_b=1.0,
                 lsgan_c=0.0,
                 hiddens_enc=[512] * 3,
                 hiddens_dec=[512] * 3,
                 hiddens_cri=[512] * 3,
                 settings=None,
                 **kwargs):
        Net.__init__(self, **kwargs)
        self._settings = settings
        self._models_names = ['ae', 'enc', 'dec', 'cri', 'gen']
        self._is_train_step = [True, True, False, True, False]

        self._latent_dims = self._update_settings('latent_dims', latent_dims)
        self._lsgan_a = self._update_settings('lsgan_a', lsgan_a)
        self._lsgan_b = self._update_settings('lsgan_b', lsgan_b)
        self._lsgan_c = self._update_settings('lsgan_c', lsgan_c)
        self._hiddens_enc = self._update_settings('hiddens_enc', hiddens_enc)
        self._hiddens_dec = self._update_settings('hiddens_dec', hiddens_dec)
        self._hiddens_cri = self._update_settings('hiddens_cri', hiddens_cri)
        self._data = None
        self._dec_out = None
        self._latent_data = None
        self._latent_sample = None
        self._logit_true = None
        self._logit_fake = None

    def _define_losses(self):
        with tf.name_scope('loss_ae'):
            if self._losses_names[0] == 'mse':
                loss_ae = tf.losses.mean_squared_error(
                    self._data, self._dec_out)
        with tf.name_scope('loss_latent'):
            if self._losses_names[3] == 'lsgan':
                self._lsa = tf.Variable(np.ones(shape=(
                    self._batch_size, 1)) * self._lsgan_a, trainable=False, name='lsgan_a', dtype=tf.float32)
                self._lsb = tf.Variable(np.ones(shape=(
                    self._batch_size, 1)) * self._lsgan_b, trainable=False, name='lagan_b', dtype=tf.float32)
                self._lsc = tf.Variable(np.ones(shape=(
                    self._batch_size, 1)) * self._lsgan_c, trainable=False, name='lsgan_c', dtype=tf.float32)
                with tf.name_scope('loss_enc'):
                    loss_enc = tf.losses.mean_squared_error(
                        self._lsc, self._logit_fake)
                with tf.name_scope('loss_cri'):
                    loss_cri = tf.losses.mean_squared_error(
                        self._lsa, self._logit_fake) + tf.losses.mean_squared_error(self._lsb, self._logit_true)

        self._losses[0] = loss_ae
        self._losses[1] = loss_enc
        self._losses[2] = None
        self._losses[3] = loss_cri

        tf.summary.scalar('loss_ae', loss_ae)
        tf.summary.scalar('loss_enc', loss_enc)
        tf.summary.scalar('loss_cri', loss_cri)

    def _define_optims(self):
        with tf.name_scope('optim_ae'):
            optim_ae = tf.train.RMSPropOptimizer(self._lrs_tensor[0])

        with tf.name_scope('optim_gen'):
            optim_gen = tf.train.RMSPropOptimizer(self._lrs_tensor[1])

        with tf.name_scope('optim_cri'):
            optim_cri = tf.train.RMSPropOptimizer(self._lrs_tensor[3])

        self._optims[0] = optim_ae
        self._optims[1] = None        
        self._optims[2] = optim_gen
        self._optims[3] = optim_cri

    def _define_train_steps(self):
        with tf.name_scope('train_ae'):
            gen_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='ae')
            self._train_steps[0] = self._optims[0].minimize(
                self._losses[0], var_list=gen_vars)
        with tf.name_scope('train_cri'):
            cri_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='cri')
            self._train_steps[1] = self._optims[1].minimize(
                self._losses[1], var_list=cri_vars)
        with tf.name_scope('train_gen'):
            gen_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')

        for i in range(self._nb_model):
            if not self._is_train_step[i]:
                tf.summary.scalar(
                    'lr_' + self._models_names[i], self._lrs_tensor[i])

    def _define_models(self):
        with tf.name_scope('data_input'):
            self._data = Input(shape=self._inputs_dims[0], name='data')
        with tf.name_scope('z_input'):
            self._z = tf.random_normal(
                shape=(self._batch_size, self._latent_dims), name='z')

        # build encoder

        with tf.name_scope('encoder'):
            x = self._data
            tf.summary.image('data', self._data, max_outputs=16)
            with tf.name_scope('flatten'):
                x = tf.reshape(x, shape=(self._batch_size, np.prod(
                    self._inputs_dims[0])), name='flatten')
            with tf.name_scope('encoder_denses'):
                for dim in self._hiddens_enc:
                    with tf.name_scope('dense'):
                        x = Dense(dim, activation='elu')(x)
            with tf.name_scope('encoder_latent'):
                x = Dense(self._latent_dims)(x)
            self._latent_data = x

        with tf.name_scope('decoder'):
            x = self._latent_data
            with tf.name_scope('decoder_denses'):
                for dim in self._hiddens_dec:
                    with tf.name_scope('dense'):
                        x = Dense(dim, activation='elu')(x)
            with tf.name_scope('decoder_generated'):
                x = Dense(np.prod(self._inputs_dims[0]))(x)
            with tf.name_scope('reshape'):
                x = tf.reshape(
                    x, shape=[self._batch_size] + list(self._inputs_dims[0]), name='reshape')
            self._dec_out = x
            tf.summary.image('decoded_data', self._dec_out, max_outputs=16)
                
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

        self._logit_true = cri(self._data)
        self._logit_fake = cri(self._generated)

        self._inputs[0] = [self._data, self._latent_input]
        self._outputs[0] = [self._logit_fake]

        self._inputs[1] = [self._data, self._latent_input]
        self._outputs[1] = [self._logit_true, self._logit_fake]

        self._inputs[2] = [self._latent_input]
        self._outputs[2] = [self._generated]

        # self._define_clip_steps()

    @property
    def clip_steps(self):
        return self._clip_step

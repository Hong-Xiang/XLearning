""" Implementation of AAE network on MNIST """

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from .base import NetGen
from ..utils.general import with_config, print_global_vars
from ..model.layers import dense_stack, Input


class AAE1D(NetGen):

    @with_config
    def __init__(self,
                 lsgan_a=-1.0,
                 lsgan_b=1.0,
                 lsgan_c=1.0,
                 hiddens_enc=[1000] * 2,
                 hiddens_dec=[1000] * 2,
                 hiddens_cri=[1000] * 2,
                 is_bn=False,
                 is_dropout=False,
                 settings=None,
                 **kwargs):
        NetGen.__init__(self, **kwargs)
        self._settings = settings
        self._models_names = ['ae', 'enc_logit', 'cri', 'enc', 'Gen']
        self._is_train_step = [True, True, True, False, False]

        self._lsgan_a = self._update_settings('lsgan_a', lsgan_a)
        self._lsgan_b = self._update_settings('lsgan_b', lsgan_b)
        self._lsgan_c = self._update_settings('lsgan_c', lsgan_c)
        self._hiddens_enc = self._update_settings('hiddens_enc', hiddens_enc)
        self._hiddens_dec = self._update_settings('hiddens_dec', hiddens_dec)
        self._hiddens_cri = self._update_settings('hiddens_cri', hiddens_cri)
        self._is_bn = self._update_settings('is_bn', is_bn)
        self._is_dropout = self._update_settings('is_dropout', is_dropout)

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
                    self._data, self._dec_data)
        with tf.name_scope('loss_latent'):
            # if self._losses_names[3] == 'lsgan':
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
        self._losses[2] = loss_cri

        tf.summary.scalar('loss_ae', loss_ae)
        tf.summary.scalar('loss_enc', loss_enc)
        tf.summary.scalar('loss_cri', loss_cri)

    def _define_optims(self):
        with tf.name_scope('optim_ae'):
            optim_ae = tf.train.RMSPropOptimizer(self._lrs_tensor[0])

        with tf.name_scope('optim_enc'):
            optim_enc = tf.train.RMSPropOptimizer(self._lrs_tensor[1])

        with tf.name_scope('optim_cri'):
            optim_cri = tf.train.RMSPropOptimizer(self._lrs_tensor[2])

        self._optims[0] = optim_ae
        self._optims[1] = optim_enc
        self._optims[2] = optim_cri

    def _define_train_steps(self):
        enc_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='enc_vars')
        dec_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='dec_vars')
        cri_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='cri_vars')
        with tf.name_scope('train_ae'):
            ae_vars = enc_vars + dec_vars
            self._train_steps[0] = self._optims[0].minimize(
                self._losses[0], var_list=ae_vars)
        with tf.name_scope('train_enc'):
            self._train_steps[1] = self._optims[1].minimize(
                self._losses[1], var_list=enc_vars)
        with tf.name_scope('train_cri'):
            self._train_steps[2] = self._optims[2].minimize(
                self._losses[2], var_list=cri_vars)

        for i in range(self._nb_model):
            if not self._is_train_step[i]:
                tf.summary.scalar(
                    'lr_' + self._models_names[i], self._lrs_tensor[i])

    def _define_models(self):
        with tf.name_scope('data_input'):
            self._data = Input(shape=(28 * 28,), name='data')
        with tf.name_scope('z_input'):
            self._z = Input(shape=(self._latent_dims,), name='z')

        # build encoder
        with tf.name_scope('enc'):
            x = self._data
            with tf.name_scope('encoder_denses'):
                x = dense_stack(x, self._latent_dims,
                                self._hiddens_enc, var_scope='enc_vars')
            self._latent_data = x

        with tf.name_scope('dec'):
            x = self._latent_data
            self._dec_data = dense_stack(
                self._latent_data, 28 * 28, self._hiddens_dec, var_scope='dec_vars')
            self._dec_z = dense_stack(
                self._z, 28 * 28, self._hiddens_dec, var_scope='dec_vars', is_reuse=True)

        with tf.name_scope('reshape'):
            img_shape = (self._batch_size, 28, 28, 1)
            data_img = tf.reshape(self._data, shape=img_shape)
            dec_data_img = tf.reshape(self._dec_data, shape=img_shape)
            dec_z_img = tf.reshape(self._dec_z, shape=img_shape)
            tf.summary.image('data', self._data)
            tf.summary.image('decoded_data', dec_data_img)
            tf.summary.image('decoded_z', dec_z_img)

        with tf.name_scope('cri'):
            with tf.name_scope('cri_denses'):
                self._logit_fake = dense_stack(
                    self._latent_data, 1, self._hiddens_cri, var_scope='cri_vars')
                self._logit_true = dense_stack(
                    self._z, 1, self._hiddens_cri, var_scope='cri_vars', is_reuse=True)

        self._inputs[0] = [self._data]
        self._outputs[0] = [self._dec_data]

        self._inputs[1] = [self._data]
        self._outputs[1] = [self._logit_fake]

        self._inputs[2] = [self._data, self._z]
        self._outputs[2] = [self._logit_fake, self._logit_true]

        self._inputs[3] = [self._data]
        self._outputs[3] = [self._latent_data]

        self._inputs[4] = [self._z]
        self._outputs[4] = [dec_z_img]


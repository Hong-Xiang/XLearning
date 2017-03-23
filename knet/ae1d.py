import numpy as np
import tensorflow as tf
from tqdm import tqdm

from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda, BatchNormalization, Activation, UpSampling2D, Convolution2D, Dropout, Flatten, LeakyReLU, GaussianNoise, Reshape, ELU, MaxPooling2D, Deconvolution2D, concatenate
from keras.optimizers import Adam, RMSprop
from keras.models import Model, Sequential
from keras import regularizers
from keras import losses
from keras.callbacks import ModelCheckpoint
import keras.backend as K


from .base import KNet, KAE, KGen
from ..dataset.mnist import mnist
from ..keras_ext.constrains import MaxMinValue
from ..kmodel.dense import dense_stack, critic1d_wgan
from ..kmodel.image import convolution_block
from ..utils.general import with_config, ProgressTimer


class AE1D(KAE):
    @with_config
    def __init__(self, **kwargs):
        super(AE1D, self).__init__(**kwargs)
        self._models_names = ['ae', 'enc', 'dec']
        self._is_trainable = [True, False, False]
        self._nb_model = 3

    def _define_models(self):
        # this is our input placeholder
        input_x = Input(shape=self._inputs_shapes[0])
        x_dim = self._inputs_shapes[0][0]
        z_dim = self._latent_dim
        y_dim = self._outputs_shapes[0][0]
        # "encoded" is the encoded representation of the input
        encoder = dense_stack(x_dim, z_dim, self._hiddens, name='enc')
        # "decoded" is the lossy reconstruction of the input
        # decoder = Dense(self._input_dim, activation='sigmoid')
        decoder = dense_stack(z_dim, y_dim, self._hiddens, name='dec')
        encoded_x = encoder(input_x)
        decoded_x = decoder(encoded_x)

        # this model maps an input to its reconstruction
        self._models[self.model_id('ae')] = Model(
            input_x, decoded_x, name='ae')
        self._models[self.model_id('enc')] = Model(
            input_x, encoded_x, name='enc')

        input_z = Input(shape=(self._latent_dim,))
        decoded_z = decoder(input_z)
        self._models[self.model_id('dec')] = Model(
            input_z, decoded_z, name='dec')


class VAE1D(KAE, KGen):
    @with_config
    def __init__(self,
                 sigma=1.0,
                 settings=None,
                 **kwargs):
        KAE.__init__(self, **kwargs)
        KGen.__init__(self, **kwargs)
        self._settings = settings
        self._sigma = self._update_settings('sigma', sigma)
        self._models_names = ['ae', 'enc', 'dec', 'gen']
        self._is_trainable = [True, False, False, False]

    def _sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(self._batch_size, self._latent_dim), mean=0.0, stddev=self._sigma)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def _vae_loss(self, x, x_decoded_mean):
        # xent_loss = losses.binary_crossentropy(x, x_decoded_mean) / self._sigma
        xent_loss = losses.mean_absolute_error(x, x_decoded_mean) / self._sigma
        # xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        # xent_loss = self._input_dim * objectives.poisson(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + self._z_log_var -
                                 K.square(self._z_mean) - K.exp(self._z_log_var), axis=-1)
        return xent_loss + 1.0 * kl_loss

    def _define_losses(self):
        for i, md_name in enumerate(self._models_names):
            if md_name == 'ae':
                self._losses[i] = self._vae_loss
            else:
                self._losses[i] = 'mse'

    def _define_models(self):
        # this is our input placeholder
        x_dim = self._inputs_shapes[0][0]
        z_dim = self._latent_dim
        y_dim = self._outputs_shapes[0][0]
        input_x = Input(shape=(x_dim,))
        with tf.name_scope('encoder'):
            with tf.name_scope('mean') as scope:
                self._z_mean = dense_stack(x_dim, z_dim, self._hiddens,
                                           name=scope)(input_x)
            with tf.name_scope('var') as scope:
                self._z_log_var = dense_stack(x_dim, z_dim, self._hiddens,
                                              name=scope)(input_x)
            with tf.name_scope('sample'):
                z = Lambda(self._sampling, output_shape=(z_dim,), name='latent')(
                    [self._z_mean, self._z_log_var])
        with tf.name_scope('decoder'):
            decoder = dense_stack(z_dim, y_dim, self._hiddens, name='dec')
        decoded_x = decoder(z)

        self._models[self.model_id('ae')] = Model(input_x, decoded_x)
        self._models[self.model_id('enc')] = Model(input_x, z)
        input_z = Input(shape=(z_dim,))
        decoded_z = decoder(input_z)
        self._models[self.model_id('dec')] = Model(input_z, decoded_z)
        self._models[self.model_id('gen')] = Model(input_z, decoded_z)


class CVAE1D(KAE, KGen):
    @with_config
    def __init__(self,
                 sigma=1.0,
                 settings=None,
                 **kwargs):
        KAE.__init__(self, **kwargs)
        KGen.__init__(self, **kwargs)
        self._settings = settings
        self._sigma = self._update_settings('sigma', sigma)
        self._models_names = ['ae', 'enc', 'dec', 'gen']
        self._is_trainable = [True, False, False, False]

    def _sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(self._batch_size, self._latent_dim), mean=0.0, stddev=self._sigma)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def _vae_loss(self, x, x_decoded_mean):
        # xent_loss = losses.binary_crossentropy(x, x_decoded_mean) / self._sigma
        xent_loss = losses.mean_squared_error(x, x_decoded_mean) / self._sigma
        # xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        # xent_loss = self._input_dim * objectives.poisson(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + self._z_log_var -
                                 K.square(self._z_mean) - K.exp(self._z_log_var), axis=-1)
        return xent_loss + 1.0 * kl_loss

    def _define_losses(self):
        for i, md_name in enumerate(self._models_names):
            if md_name == 'ae':
                self._losses[i] = self._vae_loss
            else:
                self._losses[i] = self._vae_loss

    def _define_models(self):
        # this is our input placeholder
        x_dim = self._inputs_shapes[0][0]
        c_dim = self._inputs_shapes[1][0]
        y_dim = self._outputs_shapes[0][0]
        z_dim = self._latent_dim
        input_x = Input(shape=(x_dim,))
        input_c = Input(shape=(c_dim,))
        input_m = concatenate([input_x, input_c])
        with tf.name_scope('encoder'):
            with tf.name_scope('mean'):
                self._z_mean = dense_stack(
                    x_dim + c_dim, z_dim, self._hiddens, name='enc_mean')(input_m)
            with tf.name_scope('log_var'):
                self._z_log_var = dense_stack(
                    x_dim + c_dim, z_dim, self._hiddens, name='enc_var')(input_m)
            with tf.name_scope('sample'):
                z = Lambda(self._sampling, output_shape=(z_dim,), name='latent')(
                    [self._z_mean, self._z_log_var])
        with tf.name_scope('decoder'):
            decoder = dense_stack(z_dim + c_dim, y_dim,
                                  self._hiddens, name='dec')
        z_c = concatenate([z, input_c])
        decoded_x = decoder(z_c)

        self._models[self.model_id('ae')] = Model(
            [input_x, input_c], decoded_x)
        self._models[self.model_id('enc')] = Model([input_x, input_c], z)
        input_z = Input(shape=(z_dim,))
        input_z_c = concatenate([input_z, input_c])
        decoded_z = decoder(input_z_c)
        self._models[self.model_id('dec')] = Model(
            [input_z, input_c], decoded_z)
        self._models[self.model_id('gen')] = Model(
            [input_z, input_c], decoded_z)


class CAAE1D(KAE, KGen):
    @with_config
    def __init__(self,
                 sigma=1.0,
                 settings=None,
                 **kwargs):
        KAE.__init__(self, **kwargs)
        KGen.__init__(self, **kwargs)
        self._settings = settings
        self._sigma = self._update_settings('sigma', sigma)
        self._models_names = ['ae', 'cri_t', 'cri_f', 'enc', 'dec', 'gen']
        self._is_trainable = [True, True, True, False, False, False]
        self._logit_true = tf.Variable(
            np.ones((self._batch_size, 1)), trainable=False, name='logit_true')
        self._logit_fake = tf.Variable(-np.ones((self._batch_size, 1)),
                                       trainable=False, name='logit_fake')

    def _sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(self._batch_size, self._latent_dim), mean=0.0, stddev=self._sigma)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def _vae_loss(self, x, x_decoded_mean):
        # xent_loss = losses.binary_crossentropy(x, x_decoded_mean) / self._sigma
        xent_loss = losses.mean_squared_error(x, x_decoded_mean) / self._sigma
        # xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        # xent_loss = self._input_dim * objectives.poisson(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + self._z_log_var -
                                 K.square(self._z_mean) - K.exp(self._z_log_var), axis=-1)
        return xent_loss + 1.0 * kl_loss

    def _loss_cri(self, x, logit_true):
        return K.mean(self._logit_true - self._logit_fake)

    def _loss_gen(self, x, logit_fake):
        return K.mean(-self._logit_fake)

    def _define_losses(self):
        for i, md_name in enumerate(self._models_names):
            if md_name == 'ae':
                self._losses[i] = 'mse'
            elif md_name == 'cri_t':
                self._losses[i] = self._loss_cri
            elif md_name == 'cri_f':
                self._losses[i] = self._loss_gen
            else:
                self._losses[i] = 'mse'

    def _compile_models(self):
        for i, md_name in enumerate(self._models_names):
            if md_name == 'cri_t':
                self.model('ae').trainable = False
                self.model('cri_t').compile(
                    loss=self._loss_cri, optimizer=self._optims[i])
                self.model('ae').trainable = True
            elif md_name == 'cri_f':
                self._critic.trainable = False
                self.model('cri_f').compile(
                    loss=self._loss_gen, optimizer=self._optims[i])
                self._critic.trainable = True
            else:
                self._models[i].compile(
                    loss=self._losses[i], optimizer=self._optims[i])

    def _define_models(self):
        # this is our input placeholder
        x_dim = self._inputs_shapes[0][0]
        c_dim = self._inputs_shapes[1][0]
        y_dim = self._outputs_shapes[0][0]
        z_dim = self._latent_dim
        input_x = Input(shape=(x_dim,), name='input_x')
        input_c = Input(shape=(c_dim,), name='input_c')
        input_m = concatenate([input_x, input_c])
        with tf.name_scope('encoder') as scope:
            latent = dense_stack(x_dim + c_dim, z_dim,
                                 self._hiddens, name=scope + "denses")(input_m)
        with tf.name_scope('decoder'):
            decoder = dense_stack(z_dim + c_dim, y_dim,
                                  self._hiddens, name='dec')
        z_c = concatenate([latent, input_c])
        decoded_x = decoder(z_c)
        with tf.name_scope('critic') as scope:
            critic = critic1d_wgan(
                z_dim, 1, self._hiddens, name=scope + 'critic')
            self._critic = critic
        input_z = Input(shape=(z_dim,), name='input_z')
        self._logit_true = critic(input_z)
        self._logit_fake = critic(latent)

        self._models[self.model_id('ae')] = Model(
            [input_x, input_c], decoded_x)
        self._models[self.model_id('enc')] = Model([input_x, input_c], latent)
        input_z_c = concatenate([input_z, input_c])
        decoded_z = decoder(input_z_c)
        self._models[self.model_id('dec')] = Model(
            [input_z, input_c], decoded_z)
        self._models[self.model_id('gen')] = Model(
            [input_z, input_c], decoded_z)
        self._models[self.model_id('cri_t')] = Model(
            [input_x, input_c, input_z], [self._logit_true, self._logit_fake])
        self._models[self.model_id('cri_f')] = Model(
            [input_x, input_c], self._logit_fake)

    def fit_ae(self, dataset, phase=0):
        self.model_ae.fit_generator(
            dataset, steps_per_epoch=self._steps_per_epoch, epochs=self._epochs, verbose=1,
            callbacks=[ModelCheckpoint(
                filepath=r"weightsP{0:1d}.{epoch:02d}-{loss:.5f}.hdf5".format(phase), period=1)]
        )

    def fit_p2(self, dataset, phase=0):
        tmp = np.ones((self._batch_size, 1))
        pt = ProgressTimer(100)
        for i in range(100):
            s = next(dataset)
            z = self.gen_noise()
            loss_t = self.model('cri_t').train_on_batch(
                [s[0][0], s[0][1], z], [tmp, tmp])
            loss_t = loss_t[0]
            pt.event(i, 'Pretrain, loss_t%5f'%loss_t)
        for i in range(self._epochs):
            pt = ProgressTimer(self._steps_per_epoch)
            for j in range(self._steps_per_epoch):
                s = next(dataset)
                z = self.gen_noise()
                loss_t = self.model('cri_t').train_on_batch(
                    [s[0][0], s[0][1], z], [tmp, tmp])
                loss_t = loss_t[0]
                s = next(dataset)
                loss_f = self.model('cri_f').train_on_batch(
                    [s[0][0], s[0][1]], [tmp])
                s = next(dataset)
                loss_a = self.model('ae').train_on_batch(
                    [s[0][0], s[0][1]], s[1])
                msg = r"epoch=%2d/%2d, loss_t=%5f, loss_f=%5f, loss_a=%5f"%(i, self._epochs, loss_t, loss_f, loss_a)                
                pt.event(j, msg)

""" some autoencoders on mnist
[Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
"""
import numpy as np

from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda, merge, BatchNormalization, Activation, UpSampling2D, Convolution2D, Dropout, Flatten, LeakyReLU, GaussianNoise, Reshape, ELU, MaxPooling2D, Deconvolution2D, Merge
from keras.optimizers import Adam, RMSprop
from keras.models import Model, Sequential
from keras import regularizers
from keras import objectives

import keras.backend as K


from .base import KNet, KGen
from ..dataset.mnist import mnist
from ..keras_ext.constrains import MaxMinValue
from ..kmodel.dense import denses
from ..kmodel.image import convolution_block


class AutoEncoder(KNet):

    def __init__(self, input_dim=784, encoding_dim=32, is_l1=False, l1_weight=1e-4, **kwargs):
        super(AutoEncoder, self).__init__(input_dim=input_dim, encoding_dim=encoding_dim,
                                          is_l1=is_l1, l1_weight=l1_weight, **kwargs)
        self._input_dim = self._settings['input_dim']
        self._encoding_dim = self._settings['encoding_dim']
        self._encoder = None
        self._decoder = None
        self._is_l1 = self._settings['is_l1']
        self._l1_weight = self._settings['l1_weight']
        self._nb_model = 3
    
    def get_model_id(self, name):        
        if name=='encoder':
            return 0
        if name=='decoder':
            return 1
        if name=='ae':
            return 2

    def _define_models(self):
        # this is our input placeholder
        input_img = Input(shape=(self._input_dim,))
        # "encoded" is the encoded representation of the input
        if self._is_l1:
            encoded = Dense(self._encoding_dim, activation='relu',
                            activity_regularizer=regularizers.activity_l1(self._l1_weight))(input_img)
        else:
            encoded = Dense(self._encoding_dim, activation='relu')(input_img)
        # "decoded" is the lossy reconstruction of the input
        # decoder = Dense(self._input_dim, activation='sigmoid')
        decoder = Dense(self._input_dim)
        decoded = decoder(encoded)
        # this model maps an input to its reconstruction
        self._models[self.get_model_id('ae')] = Model(input=input_img, output=decoded, name='AutoEncoder')
        self._models[self.get_model_id('encoder')] = Model(input=input_img, output=encoded, name='Encoder')
        encoded_input = Input(shape=(self._encoding_dim,))
        decoded_layer = decoder(encoded_input)
        self._models[self.get_model_id('decoder')] = Model(input=encoded_input, output=decoded_layer, name='Decoder')

    @property
    def autoencoder(self):
        return self._models[0]

    @property
    def encoder(self):
        return self._models[1]

    @property
    def decoder(self):
        return self._models[2]




class VariationalAutoEncoder(KNet):

    def __init__(self, **kwargs):
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self._inter_dim = self._hiddens[0]
        self._encoder = None
        self._decoder = None
        self._sigma = self._settings.get('sigma', 1.0)
        self._z_mean = None
        self._z_log_var = None
        self._encoding_dim = self._settings.get('encoding_dim', 1)
        self._input_dim = self._settings.get('input_dim', 28 * 28)
        self._batch_size = self._settings['batch_size']
        self._nb_model = 3

    def _sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(self._batch_size, self._encoding_dim), mean=0.0, std=self._sigma)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def _vae_loss(self, x, x_decoded_mean):
        xent_loss = self._input_dim * \
            objectives.binary_crossentropy(x, x_decoded_mean)
        # xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        # xent_loss = self._input_dim * objectives.poisson(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + self._z_log_var -
                                K.square(self._z_mean) - K.exp(self._z_log_var), axis=-1)
        return xent_loss + 1.0 * kl_loss

    def _define_losses(self):
        self._losses = [self._vae_loss] * self._nb_model

    def _define_models(self):
        # this is our input placeholder
        input_img = Input(shape=(784,))
        h = Dense(self._inter_dim)(input_img)
        self._z_mean = Dense(self._encoding_dim, name='z_mean')(h)
        self._z_log_var = Dense(self._encoding_dim, name='z_log_sigma')(h)
        z = Lambda(self._sampling, output_shape=(self._encoding_dim,), name='latent')(
            [self._z_mean, self._z_log_var])
        decoder_h = Dense(self._inter_dim, activation=self._activ)
        decoder_mean = Dense(self._input_dim, activation='sigmoid')
        h_decoded = decoder_h(z)
        x_decoded_mean = decoder_mean(h_decoded)

        self._models.append(Model(input=input_img, output=x_decoded_mean))
        self._models.append(Model(input=input_img, output=z))
        decoder_input = Input(shape=(self._encoding_dim,))
        _h_decoded = decoder_h(decoder_input)
        _x_decoded_mean = decoder_mean(_h_decoded)
        self._models.append(Model(decoder_input, _x_decoded_mean))

    @property
    def autoencoder(self):
        return self._models[0]

    @property
    def encoder(self):
        return self._models[1]

    @property
    def decoder(self):
        return self._models[2]


class GAN(KNet):
    """based on https://github.com/osh/KerasGAN/blob/master/MNIST_CNN_GAN_v2.ipynb """

    def __init__(self, **kwargs):
        super(GAN, self).__init__(**kwargs)
        self._gen = None
        self._dis = None
        self._gan = None
        self._encoding_dim = self._settings.get(
            'encoding_dim', 128)
        self._nb_model = 3
        self.dopt = None
        self.opt = None

    def _define_losses(self):
        pass

    def _define_optims(self):
        """ optimizer """
        pass

    def gen_latent(self):
        return np.random.uniform(size=(self._batch_size, self._encoding_dim))

    def gen_data(self):
        return self._gen.predict(self.gen_latent())

    def prepare_data(self, x, gen_imgs=None):
        n_image = x.shape[0]
        if gen_imgs is None:
            gen_imgs = self.gen_data()
        datas = np.concatenate((x, gen_imgs))

        y = np.zeros((n_image * 2, 2))
        y[:n_image, 0] = 1
        y[n_image:, 1] = 1
        return (datas, y)

    def _define_models(self):
        opt = Adam(1e-3)
        dopt = Adam(1e-4)
        self.dopt = dopt
        self.opt = opt
        nch = self._hiddens[0]
        g_input = Input(shape=[self._encoding_dim])
        H = Dense(nch * 14 * 14, init=self._init)(g_input)
        H = BatchNormalization(mode=2)(H)
        H = Activation('relu')(H)
        H = Reshape([14, 14, nch])(H)
        H = UpSampling2D(size=(2, 2))(H)
        H = Convolution2D(nch, 3, 3, border_mode='same',
                          init=self._init)(H)
        H = BatchNormalization(mode=2)(H)
        H = Activation('relu')(H)
        H = Convolution2D(nch // 4, 3, 3, border_mode='same',
                          init=self._init)(H)
        H = BatchNormalization(mode=2)(H)
        H = Activation('relu')(H)
        H = Convolution2D(1, 1, 1, border_mode='same', init=self._init)(H)
        g_V = Activation('sigmoid')(H)
        generator = Model(g_input, g_V)
        generator.compile(loss='binary_crossentropy', optimizer=opt)
        generator.summary()
        self._gen = generator
        self._models.append(generator)

        # Build Discriminative model ...
        d_input = Input(shape=(28, 28, 1))
        H = Convolution2D(256, 5, 5, subsample=(
            2, 2), border_mode='same', activation='relu')(d_input)
        H = LeakyReLU(0.2)(H)
        H = Dropout(self._dropout_rate)(H)
        H = Convolution2D(512, 5, 5, subsample=(
            2, 2), border_mode='same', activation='relu')(H)
        H = LeakyReLU(0.2)(H)
        H = Dropout(self._dropout_rate)(H)
        H = Flatten()(H)
        H = Dense(256)(H)
        H = LeakyReLU(0.2)(H)
        H = Dropout(self._dropout_rate)(H)
        d_V = Dense(2, activation='softmax')(H)
        discriminator = Model(d_input, d_V)
        discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
        discriminator.summary()
        self._models.append(discriminator)
        self._dis = discriminator

        # Freeze weights in the discriminator for stacked training
        def make_trainable(net, val):
            net.trainable = val
            for l in net.layers:
                l.trainable = val
        make_trainable(discriminator, False)
        # Build stacked GAN model
        gan_input = Input(shape=[self._encoding_dim])
        H = generator(gan_input)
        gan_V = discriminator(H)
        GAN = Model(gan_input, gan_V)
        GAN.compile(loss='categorical_crossentropy', optimizer=opt)
        GAN.summary()
        self._models.append(GAN)
        self._gan = GAN

    @property
    def model_dis(self):
        return self._models[1]

    @property
    def model_gen(self):
        return self._models[0]

    @property
    def model_GAN(self):
        return self._models[2]


class WGAN_old(KNet):

    def __init__(self, **kwargs):
        super(WGAN, self).__init__(**kwargs)
        self._gen = None
        self._cri = None
        self._wgan = None
        self._encoding_dim = self._settings.get(
            'encoding_dim', (128,))
        self._input_dim = self._settings.get('input_dim', (28, 28, 1))
        self._nch = self._settings.get('nch', 128)
        self._nb_model = 4

    def get_model_id(self, name):
        if name == 'cri':
            return 0
        if name == 'gen':
            return 1
        if name == 'wgan_cri':
            return 2
        if name == 'wgan_gen':
            return 3

    def _define_losses(self):
        for i in range(self._nb_model):
            self._losses[i] = ['mse']
        self._losses[self.get_model_id('wgan_gen')] = self._g_loss
        self._losses[self.get_model_id('wgan_cri')] = self._c_loss

    def gen_latent(self):
        return np.random.uniform(low=-1.0, high=1.0, size=(self._batch_size, self._encoding_dim))

    def gen_data(self):
        return self.gen.predict(self.gen_latent())

    def _c_loss(self, label, predict):
        return K.sum(self._f_logit - self._t_logit, axis=-1)

    def _g_loss(self, label, predict):
        return K.sum(- self._f_logit, axis=-1)

    def _compile_models(self):
        i_cri = self.get_model_id('cri')
        i_gen = self.get_model_id('gen')
        i_wgan_cri = self.get_model_id('wgan_cri')
        i_wgan_gen = self.get_model_id('wgan_gen')
        self.cri.compile(loss=self._losses[
                         i_cri], optimizer=self._optims[i_cri])
        self.gen.compile(loss=self._losses[
                         i_gen], optimizer=self._optims[i_gen])
        self.cri.summary()
        self.gen.summary()
        self.gen.trainable = False
        self.wgan_cri.compile(
            loss=self._losses[i_wgan_cri], optimizer=self._optims[i_wgan_cri])
        self.wgan_cri.summary()
        self.gen.trainable = True
        self.cri.trainable = False
        self.wgan_gen.compile(
            loss=self._losses[i_wgan_gen], optimizer=self._optims[i_wgan_gen])
        self.wgan_gen.summary()

    def _define_models(self):
        nch = self._hiddens[0]
        g_input = Input(shape=(self._encoding_dim,), name='g_input')

        # Build Gen model
        h = Dense(nch * 14 * 14, W_constraint=MaxMinValue(),
                  b_constraint=MaxMinValue())(g_input)
        h = BatchNormalization(mode=2)(h)
        h = Activation('relu')(h)
        h = Reshape([14, 14, nch])(h)
        h = UpSampling2D(size=(2, 2))(h)
        h = Convolution2D(nch, 3, 3, border_mode='same',
                          W_constraint=MaxMinValue(), b_constraint=MaxMinValue())(h)
        h = BatchNormalization(mode=2)(h)
        h = Activation('relu')(h)
        h = Convolution2D(nch // 4, 3, 3, border_mode='same',
                          W_constraint=MaxMinValue(), b_constraint=MaxMinValue())(h)
        h = BatchNormalization(mode=2)(h)
        h = Activation('relu')(h)
        g_V = Convolution2D(1, 1, 1, border_mode='same', W_constraint=MaxMinValue(
        ), b_constraint=MaxMinValue())(h)

        self._models[self.get_model_id('gen')] = Model(
            g_input, g_V, name='Gen')

        # Build Cri model ...
        t_input = Input(shape=self._input_dim, name='t_input')
        c = Sequential(name='Cri_Seq')
        c.add(Convolution2D(256, 5, 5, subsample=(
            2, 2), border_mode='same', activation='relu', input_shape=(28, 28, 1), W_constraint=MaxMinValue(), b_constraint=MaxMinValue()))
        c.add(Dropout(self._dropout_rate))
        c.add(Convolution2D(512, 5, 5, subsample=(
            2, 2), border_mode='same', activation='relu', W_constraint=MaxMinValue(), b_constraint=MaxMinValue()))
        c.add(Dropout(self._dropout_rate))
        c.add(Flatten())
        c.add(Dense(256, W_constraint=MaxMinValue(), b_constraint=MaxMinValue()))
        c.add(ELU())
        c.add(Dropout(self._dropout_rate))
        c.add(Dense(1, W_constraint=MaxMinValue(), b_constraint=MaxMinValue()))

        self._t_logit = c(t_input)
        self._f_logit = c(g_V)

        self._models[self.get_model_id('cri')] = Model(
            input=t_input, output=self._t_logit, name='Crix2_full')
        self._models[self.get_model_id('wgan_cri')] = Model(input=[t_input, g_input], output=[
            self._t_logit, self._f_logit], name='wgan_cri')
        self._models[self.get_model_id('wgan_gen')] = Model(
            input=g_input, output=self._f_logit, name='wgan_gen')

    @property
    def gen(self):
        return self._models[self.get_model_id('gen')]

    @property
    def cri(self):
        return self._models[self.get_model_id('cri')]

    @property
    def wgan_gen(self):
        return self._models[self.get_model_id('wgan_gen')]

    @property
    def wgan_cri(self):
        return self._models[self.get_model_id('wgan_cri')]


class WGAN(KGen):

    def __init__(self, **kwargs):
        super(WGAN, self).__init__(**kwargs)
        self._gen = None
        self._cri = None
        self._wgan = None
        self._encoding_dim = self._settings.get(
            'encoding_dim', (128,))
        self._input_dim = self._settings.get('input_dim', (28, 28, 1))
        self._clip_min = self._settings.get('clip_min', -0.01)
        self._clip_max = self._settings.get('clip_max', 0.01)
        self._hiddens_g = self._settings['hiddens_g']
        self._hiddens_c = self._settings['hiddens_c']
        self._nb_model = 3

    def get_model_id(self, name):
        if name == 'cri':
            return 0
        if name == 'gen':
            return 1
        if name == 'wgan':
            return 2

    def _define_losses(self):
        for i in range(len(self._losses)):
            self._losses[i] = self._w_loss

    def _w_loss(self, label, predict):
        return K.mean(label * predict)

    def _compile_models(self):
        i_cri = self.get_model_id('cri')
        i_gen = self.get_model_id('gen')
        i_wgan = self.get_model_id('wgan')
        self.cri.compile(loss=self._losses[
                         i_cri], optimizer=self._optims[i_cri])
        self.gen.compile(loss=self._losses[
                         i_gen], optimizer=self._optims[i_gen])
        self.cri.summary()
        self.gen.summary()
        self.cri.trainable = False
        self.wgan.compile(
            loss=self._losses[i_wgan], optimizer=self._optims[i_wgan])
        self.wgan.summary()
        self.cri.trainable = True

    def clip_theta(self):
        for l in self.cri.layers:
            weights = l.get_weights()
            weights = [np.clip(w, self._clip_min, self._clip_max)
                       for w in weights]
            l.set_weights(weights)

    def _define_models(self):
        g_input = Input(shape=(self._encoding_dim,), name='g_input')

        # Build Gen model
        gen = Dense(7 * 7 * 512)(g_input)
        gen = BatchNormalization()(gen)
        gen = ELU()(gen)
        gen = Reshape((7, 7, 512))(gen)

        gen = UpSampling2D()(gen)

        gen = Convolution2D(self._hiddens_g[0], 3, 3, border_mode='same')(gen)
        gen = BatchNormalization()(gen)
        gen = ELU()(gen)

        gen = UpSampling2D()(gen)
        gen = Convolution2D(self._hiddens_g[1], 3, 3, border_mode='same')(gen)
        gen = BatchNormalization()(gen)
        gen = ELU()(gen)

        gen = UpSampling2D()(gen)
        gen = Convolution2D(self._hiddens_g[2], 3, 3, border_mode='same')(gen)
        gen = BatchNormalization()(gen)
        gen = ELU()(gen)

        gen = Convolution2D(self._hiddens_g[3], 3, 3, subsample=(
            2, 2), border_mode='same')(gen)
        gen = BatchNormalization()(gen)
        gen = ELU()(gen)

        g_output = Convolution2D(1, 1, 1)(gen)

        self._models[self.get_model_id('gen')] = Model(
            g_input, g_output, name='Gen')
        self._model_gen = self._models[self.get_model_id('gen')]
        # Build Cri model ...
        t_input = Input(shape=self._input_dim, name='c_input')

        c_conv = []
        c_bn = []
        c_elu = []
        c_maxp = []
        for i in range(len(self._hiddens_c)):
            if i == 0:
                c_conv.append(Convolution2D(self._hiddens_c[
                              i], 3, 3, input_shape=self._input_dim, W_constraint=MaxMinValue(), name='cri_conv_%d' % i))
            else:
                c_conv.append(Convolution2D(self._hiddens_c[
                              i], 3, 3, W_constraint=MaxMinValue(), name='cri_conv_%d' % i))
            c_maxp.append(MaxPooling2D(name='cri_maxp_%d' % i))
            c_bn.append(BatchNormalization(name='cri_bn_%d' % i))
            c_elu.append(ELU(name='cri_elu_%d' % i))
        c_layers = []
        for l_conv, l_bn, l_elu, l_maxp in zip(c_conv, c_bn, c_elu, c_maxp):
            c_layers.append(l_conv)
            c_layers.append(l_bn)
            c_layers.append(l_elu)
            c_layers.append(l_maxp)

        c_layers.append(Flatten())
        c_layers.append(Dense(256))
        c_layers.append(Dense(1))
        cri = Sequential(layers=c_layers, name='Cri')

        t_logit = cri(t_input)
        f_logit = cri(g_output)

        self._models[self.get_model_id('cri')] = Model(
            input=t_input, output=t_logit, name='Cri')
        self._models[self.get_model_id('wgan')] = Model(
            input=g_input, output=f_logit, name='wgan')

    @property
    def gen(self):
        return self._models[self.get_model_id('gen')]

    @property
    def cri(self):
        return self._models[self.get_model_id('cri')]

    @property
    def wgan(self):
        return self._models[self.get_model_id('wgan')]


class VAE(KGen):

    def __init__(self, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self._sigma = self._settings.get('sigma', 1.0)
        
        self._nb_model = 3

    def _sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(self._batch_size, self._encoding_dim), mean=0.0, std=1.0)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def _vae_loss(self, x, p):
        xent_loss = objectives.mse(x, p) / self._sigma
        # xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        # xent_loss = self._input_dim * objectives.poisson(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(- self._encoding_dim + self._z_log_var -
                                 K.square(self._z_mean) - K.exp(self._z_log_var), axis=-1)
        return xent_loss + kl_loss

    def _define_losses(self):
        self._losses = [self._vae_loss] * self._nb_model

    def _define_models(self):
        ip = Input(shape=self._input_dim)
        phi = denses(ip, hiddens=self._hiddens, is_dropout=False,
                     dropout_rate=self._dropout_rate, name='encoder')
        self._z_mean = Dense(self._encoding_dim, name='latent_mean')(phi)
        self._z_log_var = Dense(self._encoding_dim, name='latent_var')(phi)
        z_sampled = Lambda(self._sampling, output_shape=(
            self._encoding_dim,), name='repa')([self._z_mean, self._z_log_var])
        theta = denses(input_shape=(self._encoding_dim, ), is_dropout=False,
                       hiddens=self._hiddens, name='decoder')
        hg = Dense(self._input_dim[0])
        y = hg(theta(z_sampled))
        z_input = Input(shape=(self._encoding_dim,))
        yg = hg(theta(z_input))

        self._models[0] = Model(input=ip, output=y)
        self._models[1] = Model(input=z_input, output=yg)
        self._models[2] = Model(input=ip, output=self._z_mean)
        self._model_gen = self._models[1]

        @property
        def vae(self):
            return self._models[0]


class CVAE(KGen):

    def __init__(self, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self._sigma = self._settings.get('sigma', 1.0)
        self._alpha = self._settings.get('alpha', 10.0)
        self._nb_model = 2

    def _sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(self._batch_size, self._encoding_dim), mean=0.0, std=1.0)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def _vae_loss(self, x, p):
        xent_loss = objectives.mse(x, p) / self._sigma
        # xent_loss = objectives.binary_crossentropy(x, x_decoded_mean)
        xent_loss = objectives.poisson(x / self._sigma, p / self._sigma) * self._alpha
        kl_loss = - 0.5 * K.mean(- self._encoding_dim + self._z_log_var -
                                 K.square(self._z_mean) - K.exp(self._z_log_var), axis=-1)
        return xent_loss + 1.0*kl_loss

    def _define_losses(self):
        self._losses = [self._vae_loss] * self._nb_model

    def _abs(self, x):
        return K.abs(x)

    def _define_models(self):
        ip_clean = Input(shape=self._input_dim, name='clean_image')
        ip_noise = Input(shape=self._input_dim, name='noise_image')
        phi_clean = denses(ip_clean, hiddens=self._hiddens, is_dropout=False,
                           dropout_rate=self._dropout_rate, name='encoder_clean')
        phi_noise = denses(ip_noise, hiddens=self._hiddens, is_dropout=False,
                           dropout_rate=self._dropout_rate, name='encoder_noise')
        phi = merge([phi_clean, phi_noise], mode='concat', concat_axis=1)

        self._z_mean = Dense(self._encoding_dim, name='latent_mean')(phi)
        self._z_log_var = Dense(self._encoding_dim, name='latent_var')(phi)
        z_sampled = Lambda(self._sampling, output_shape=(
            self._encoding_dim,), name='latent_sample')([self._z_mean, self._z_log_var])
        theta_sample = denses(input_shape=(self._encoding_dim, ), is_dropout=False,
                              hiddens=self._hiddens, name='decoder_latent')
        theta_noise = denses(input_shape=self._input_dim,
                             hidden=self._hiddens, name='decoder_noise')

        hg = Dense(self._input_dim[0])
        y = hg(
            merge([theta_sample(z_sampled), theta_noise(ip_noise)], mode='concat'))
        # y = Lambda(self._abs)(y)
        z_input = Input(shape=(self._encoding_dim,), name='latent_input')
        yg = hg(
            merge([theta_sample(z_input), theta_noise(ip_noise)], mode='concat'))
        # yg = Lambda(self._abs)(yg)
        self._models[0] = Model(input=[ip_noise, ip_clean], output=y)
        self._models[1] = Model(input=[ip_noise, z_input], output=yg)
        self._model_gen = self._models[1]

    @property
    def vae(self):
        return self._models[0]

    def gen_data(self, noise_data):
        z_sample = self.gen_noise()
        return self._model_gen.predict([noise_data, z_sample])

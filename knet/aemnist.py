""" some autoencoders on mnist
[Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
"""
import numpy as np

from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda, merge, BatchNormalization, Activation, UpSampling2D, Convolution2D, Dropout, Flatten, LeakyReLU, GaussianNoise, Reshape
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers
from keras import objectives
import keras.backend as K

from .base import KNet
from ..dataset.mnist import mnist


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
        decoder = Dense(self._input_dim, activation='sigmoid')
        decoded = decoder(encoded)
        # this model maps an input to its reconstruction
        self._models.append(
            Model(input=input_img, output=decoded, name='Encoder'))
        self._models.append(
            Model(input=input_img, output=encoded, name='AutoEncoder'))
        encoded_input = Input(shape=(self._encoding_dim,))
        decoded_layer = decoder(encoded_input)
        self._models.append(
            Model(input=encoded_input, output=decoded_layer, name='Decoder'))

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

    def _define_losses(self):
        pass

    def _define_optims(self):
        """ optimizer """
        pass

    def prepare_data(self, x):
        n_image = x.shape[0]
        noise_input = np.random.uniform(size=(n_image, self._encoding_dim))
        gen_imgs = self._gen.predict(noise_input)
        datas = np.concatenate((x, gen_imgs))
        y = np.zeros((n_image * 2, 2))
        y[:n_image, 0] = 1
        y[n_image:, 1] = 1
        return (datas, y)

    def _define_model(self):
        opt = Adam(1e-3)
        dopt = Adam(1e-4)
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
        H = Convolution2D(nch / 4, 3, 3, border_mode='same',
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
        d_input = Input(shape=(28, 28))
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

        # Freeze weights in the discriminator for stacked training
        def make_trainable(net, val):
            net.trainable = val
            for l in net.layers:
                l.trainable = val
        make_trainable(discriminator, False)
        # Build stacked GAN model
        gan_input = Input(shape=[100])
        H = generator(gan_input)
        gan_V = discriminator(H)
        GAN = Model(gan_input, gan_V)
        GAN.compile(loss='categorical_crossentropy', optimizer=opt)
        GAN.summary()
        self._models.append(GAN)

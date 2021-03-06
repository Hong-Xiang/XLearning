import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from ..model.layers import Denses
from ..model.losses import loss_lsgan
from ..utils.general import with_config
from ..dataset.mnist import MNIST


def lsgan(input_shape, latent_dim, hiddens_enc, hiddens_dec, hiddens_cri, batch_size, lrae=1e-3, lrgan=1e-1):
    data = Input(shape=input_shape)
    z = Input(shape=(latent_dim,))
    encoder = Denses(input_shape, latent_dim, hiddens_enc)
    decoder = Denses((latent_dim,), np.prod(input_shape), hiddens_dec)
    critic = Denses((latent_dim), 1, hiddens_cri)
    latent = encoder(data)
    dec_out = decoder(latent)
    dec_z = decoder(z)
    logit_true = critic(latent)
    logit_fake = critic(z)

    optim_ae = RMSprop(lrae)
    optim_gan = RMSprop(lrgan)

    m_enc = Model(data, latent, name='encoder')
    m_dec = Model(latent, dec_out, name='decoder')
    m_gen = Model(z, logit_fake, name='gen')
    m_cri = Model([data, z], [logit_true, logit_fake], name='cri')
    m_sam = Model(z, dec_z, name='sam')
    m_ae = Model(data, dec_out, name='ae')
    m_ae.compile(optim_ae, 'mse')
    critic.trainable = False
    m_gen.compile(optim_gan, 'mse')
    critic.trainable = True
    encoder.trainable = False
    m_cri.compile(optim_gan, 'mse')
    m_enc.compile(optim_ae, 'mse')
    m_dec.compile(optim_ae, 'mse')
    m_sam.compile(optim_ae, 'mse')
    return m_ae, m_enc, m_dec, m_gen, m_cri, m_sam


def train_on_batch(data_generator, batch_size, latent_dim, m_ae, m_gen, m_cri, step_ae=1, step_gen=1, step_cri=1):
    a = tf.Variable(np.ones((batch_size, 1)) * -1.0, False)
    b = tf.Variable(np.ones((batch_size, 1)) * 1.0, False)
    c = tf.Variable(np.ones((batch_size, 1)) * 0.0, False)
    for i in range(step_ae):
        s = next(data_generator)
        loss_v = m_ae.train_on_batch(s[0], s[1])
        print('loss_ae', loss_v)
    for i in range(step_cri):
        s = next(data_generator)
        z = np.random.normal(size=(batch_size, latent_dim))
        loss_v = m_cri.train_on_batch([s[0], z], [b, a])
        print('loss_cri', loss_v)
    for i in range(step_gen):
        s = next(data_generator)
        z = np.random.normal(size=(batch_size, latent_dim))
        loss_v = m_gen.train_on_batch([z], [c])
        print('loss_gen', loss_v)


@with_config
def get_conf(filenames=None, settings=None, **kwargs):
    return settings


def main():
    c = get_conf()

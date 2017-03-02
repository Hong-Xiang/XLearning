import tensorflow as tf
import numpy as np
from keras.layers import Dense, Convolution2D, BatchNormalization, ELU, Flatten, Deconvolution2D, Reshape
from keras.objectives import mean_squared_error
from .base import Net, NetGen
from ..utils.general import with_config
from ..model.layers import Input, Label
from ..keras_ext.constrains import MaxMinValue


class WGAN1D(NetGen):

    @with_config
    def __init__(self,
                 latent_dims=None,
                 settings=None,
                 **kwargs):
        NetGen.__init__(self, **kwargs)
        self._settings = settings
        self._models_names = ['WGan', 'Cri', 'Gen']
        self._is_train_step = [True, True, False]
        self._latent_dims = self._update_settings('latent_dims', latent_dims)

    def _define_losses(self):
        with tf.name_scope('loss_gen'):
            loss_gen = tf.reduce_sum(self._logit_fake)
        with tf.name_scope('loss_cri'):
            loss_cri = tf.reduce_sum(self._logit_true) - \
                tf.reduce_sum(self._logit_fake)

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
        with tf.name_scope('train_gen'):
            gen_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='gen')
            self._train_steps[0] = self._optims[0].minimize(
                self._losses[0], var_list=gen_vars)
        with tf.name_scope('train_cri'):
            cri_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='cri')
            self._train_steps[1] = self._optims[1].minimize(
                self._losses[1], var_list=cri_vars)

    def _define_models(self):
        with tf.name_scope('input'):
            self._data = Input(shape=self._inputs_dims[0], name='data_input')
        with tf.name_scope('latent_input'):
            self._latent_input = Input(shape=self._latent_dims)
        with tf.name_scope('label'):
            self._labels = Input(shape=(None, 1), name='label')
        self._weights = []

        x = self._latent_input
        with tf.name_scope('gen'):
            with tf.name_scope('denses'):
                for dim in self._hiddens:
                    ly = Dense(dim, activation='elu')
                    x = ly(x)
            with tf.name_scope('recon'):
                ly = Dense(np.prod(self._inputs_dims))
                self._image = ly(x)

        lys = []
        with tf.name_scope('cri'):
            with tf.name_scope('reshape'):
                lys.append(Reshape((28, 28, 1)))
            with tf.name_scope('convs'):
                for dim in self._hiddens:
                    with tf.name_scope('features'):
                        lys.append(Convolution2D(
                            dim, 3, 3, activation='elu', W_constraint=MaxMinValue()))
            with tf.name_scope('flatten'):
                lys.append(Flatten())
            with tf.name_scope('denses'):
                for dim in self._hiddens:
                    lys.append(Dense(dim, activation='elu',
                                     W_constraint=MaxMinValue()))
            with tf.name_scope('logits'):
                lys.append(Dense(1, activation='sigmoid',
                                 W_constraint=MaxMinValue()))

        x = self._data
        for ly in lys:
            x = ly(x)
        self._logit_true = x

        x = self._latent_input
        for ly in lys:
            x = ly(x)
        self._logit_fake = x

        self._inputs[0] = [self._latent_input]
        self._outputs[0] = [self._logit_true]

        self._inputs[1] = [self._data, self._latent_input]
        self._outputs[1] = [self._logit_true, self._logit_fake]

        self._inputs[2] = [self._latent_input]
        self._outputs[2] = [self._image]

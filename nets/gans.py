import tensorflow as tf
import numpy as np
from keras.layers import Dense, Convolution2D, BatchNormalization, ELU, Flatten, Deconvolution2D, Reshape, UpSampling2D, Dropout
from keras.objectives import mean_squared_error
from keras.models import Sequential
from .base import Net, NetGen
from ..utils.general import with_config
from ..model.layers import Input, Label
from ..keras_ext.constrains import MaxMinValue


class WGAN(NetGen):

    @with_config
    def __init__(self,                 
                 settings=None,                 
                 **kwargs):
        NetGen.__init__(self, **kwargs)
        self._settings = settings
        self._models_names = ['WGan', 'Cri', 'Gen']
        self._is_train_step = [True, True, False]        

    def _define_losses(self):
        with tf.name_scope('loss_gen'):
            loss_gen = tf.reduce_mean(self._logit_fake)
        with tf.name_scope('loss_cri'):
            loss_cri = tf.reduce_mean(self._logit_true) - \
                tf.reduce_mean(self._logit_fake)

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

    def _define_models(self):
        with tf.name_scope('input'):
            self._data = Input(shape=self._inputs_dims[0], name='data_input')
        with tf.name_scope('latent_input'):
            self._latent_input = Input(shape=(self._latent_dims,))
        self._weights = []

        # build generator

        x = self._latent_input
        with tf.name_scope('gen'):
            with tf.name_scope('denses'):
                x = Dense(1024, activation='elu')(x)
                x = Dense(128 * 7 * 7, activation='elu')(x)
            with tf.name_scope('reshape'):
                x = Reshape((7, 7, self._batch_size))(x)
            with tf.name_scope('upsampling0'):
                x = UpSampling2D(size=(2, 2))(x)
                x = Convolution2D(256, 5, 5, border_mode='same',
                                  activation='elu', init='glorot_normal')(x)
            with tf.name_scope('upsampling1'):
                x = UpSampling2D(size=(2, 2))(x)
                x = Convolution2D(128, 5, 5, border_mode='same',
                                  activation='elu', init='glorot_normal')(x)
            with tf.name_scope('generation'):
                x = Convolution2D(1, 2, 2, border_mode='same',
                                  init='glorot_normal')(x)
                self._generated = x
                tf.summary.image('generated', self._generated, max_outputs=16)

        # build discriminator

        # lys = []
        # with tf.name_scope('cri'):

        #     with tf.name_scope('reshape'):
        #         lys.append(Reshape((28, 28, 1)))
        #     with tf.name_scope('conv_0'):
        #         lys.append(Convolution2D(32, 3, 3, subsample=(2, 2),
        #                                  activation='elu', W_constraint=MaxMinValue()))
        #         lys.append(Dropout(0.3))
        #     with tf.name_scope('conv_1'):
        #         lys.append(Convolution2D(64, 3, 3, activation='elu',
        #                                  W_constraint=MaxMinValue()))
        #         lys.append(Dropout(0.3))
        #     with tf.name_scope('conv_2'):
        #         lys.append(Convolution2D(128, 3, 3, subsample=(2, 2),
        #                                  activation='elu', W_constraint=MaxMinValue()))
        #         lys.append(Dropout(0.3))
        #     with tf.name_scope('conv_3'):
        #         lys.append(Convolution2D(
        #             256, 3, 3, activation='elu', W_constraint=MaxMinValue()))
        #         lys.append(Dropout(0.3))
        #     with tf.name_scope('flatten'):
        #         lys.append(Flatten())
        #     # with tf.name_scope('denses'):
        #     #     for dim in self._hiddens:
        #     #         lys.append(Dense(dim, activation='elu',
        #     #                          W_constraint=MaxMinValue()))
        #     with tf.name_scope('logits'):
        #         # lys.append(Dense(1, activation='sigmoid',
        #         #                  W_constraint=MaxMinValue()))
        #         lys.append(Dense(1))

        with tf.name_scope('cri'):
            cri = Sequential()
            with tf.name_scope('conv_0'):
                cri.add(Convolution2D(32, 3, 3, subsample=(2, 2),
                                      activation='elu', W_constraint=MaxMinValue(), input_shape=(28, 28, 1)))
                cri.add(Dropout(0.3))
            with tf.name_scope('conv_1'):
                cri.add(Convolution2D(64, 3, 3, activation='elu',
                                      W_constraint=MaxMinValue()))
                cri.add(Dropout(0.3))
            with tf.name_scope('conv_2'):
                cri.add(Convolution2D(128, 3, 3, subsample=(2, 2),
                                      activation='elu', W_constraint=MaxMinValue()))
                cri.add(Dropout(0.3))
            with tf.name_scope('conv_3'):
                cri.add(Convolution2D(
                    256, 3, 3, activation='elu', W_constraint=MaxMinValue()))
                cri.add(Dropout(0.3))
            with tf.name_scope('flatten'):
                cri.add(Flatten())
            with tf.name_scope('logits'):
                cri.add(Dense(1, W_constraint=MaxMinValue()))

        self._logit_true = cri(self._data)
        self._logit_fake = cri(self._generated)
        # nb_lys = len(lys)
        # x = self._data

        # for i in range(nb_lys):
        #     # DEBUG
        #     print(lys[i])
        #     print(x.get_shape())
        #     # DBUG
        #     x = lys[i](x)
        # self._logit_true = x

        # x = self._latent_input
        # for i in range(nb_lys):
        #     x = lys[i](x)
        # self._logit_fake = x

        self._inputs[0] = [self._data, self._latent_input]
        self._outputs[0] = [self._logit_fake]

        self._inputs[1] = [self._data, self._latent_input]
        self._outputs[1] = [self._logit_true, self._logit_fake]

        self._inputs[2] = [self._latent_input]
        self._outputs[2] = [self._generated]

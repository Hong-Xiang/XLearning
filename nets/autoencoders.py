import tensorflow as tf
from keras.layers import Dense, Convolution2D, BatchNormalization, ELU, Flatten, Deconvolution2D, Reshape
from keras.objectives import mean_squared_error, binary_crossentropy
from .base import Net, NetGen
from ..utils.general import with_config
from ..model.layers import Input, Label


class AutoEncoderBase(Net):

    @with_config
    def __init__(self,
                 code_dims=2,
                 settings=None,
                 **kwargs):
        super(AutoEncoderBase, self).__init__(**kwargs)
        self._settings = settings
        self._latent_dim = self._update_settings('latent_dims', code_dims)
        self._models_names = self._update_settings(
            'models_names', ['AutoEncoder', 'Encoder', 'Decoder'])
        self._is_train_step = [True, False, False]
        self._data = None
        self._label = None
        self._data_latent = None
        self._data_recon = None
        self._latent = None
        self._latent_recon = None

    def _define_losses(self):
        with tf.name_scope('loss'):
            loss_v = tf.losses.mean_squared_error(
                self._label, self._data_recon)
        self._losses[0] = loss_v
        tf.summary.scalar(name='mse_loss', tensor=loss_v)

    def _link_tensors(self):
        self._labels[0] = [self._label]
        self._inputs[0] = [self._data]
        self._outputs[0] = [self._data_recon]
        self._inputs[1] = [self._data]
        self._outputs[1] = [self._data_latent]
        self._inputs[2] = [self._latent]
        self._outputs[2] = [self._latent_recon]


class AutoEncoder1D(AutoEncoderBase):
    """ Auto encoder with a vector as input """

    @with_config
    def __init__(self, model_names=None, **kwargs):

        super(AutoEncoder1D, self).__init__(**kwargs)

    def _define_models(self):
        with tf.name_scope('data_input'):
            self._data = Input(shape=self._inputs_dims[0], name='data_input')
        with tf.name_scope('data_label'):
            self._label = Label(shape=self._inputs_dims[0])
        with tf.name_scope('encoder'):
            x = self._data
            cid = 0
            for dim in self._hiddens:
                with tf.name_scope('hidden_%d' % cid):
                    x = Dense(dim, name='encoder_inter', activation='elu')(x)
                    cid += 1
            self._data_latent = Dense(
                self._latent_dim, name='encoder', activation='elu')(x)
        with tf.name_scope('code_input'):
            self._latent = Input((self._latent_dim,), name='code_input')
        with tf.name_scope('decoder'):
            decoder_inter = []
            for dim in self._hiddens:
                decoder_inter.append(
                    Dense(dim, name='decoder_inter', activation='elu'))
            decoder_inter.append(
                Dense(self._inputs_dims[0][0], name='decoder', activation='elu'))
            x_data = self._data_latent
            x_latent = self._latent
            for ly in decoder_inter:
                x_data = ly(x_data)
                x_latent = ly(x_latent)
            self._data_recon = x_data
            self._latent_recon = x_latent
        self._link_tensors()


class VAE1D(AutoEncoderBase, NetGen):

    @with_config
    def __init__(self,
                 sigma=0.1,
                 settings=None,
                 **kwargs):
        NetGen.__init__(self, **kwargs)
        AutoEncoderBase.__init__(self, **kwargs)
        self._settings = settings
        self._sigma = self._update_settings('sigma', sigma)

    def _define_losses(self):
        with tf.name_scope('loss'):
            if self._losses_names[0] == "mse":
                xent_loss = tf.losses.mean_squared_error(
                    self._label, self._data_recon) / self._sigma
            elif self._losses_names[0] == "xetp":
                xent_loss = tf.reduce_mean(binary_crossentropy(
                    self._label, self._data_recon)) / self._sigma
            kl_loss = tf.reduce_mean(- 0.5 * (- self._latent_dim + self._latent_log_var - tf.square(
                self._latent_mean) - tf.exp(self._latent_log_var)), name='kl_loss')
            self._losses[0] = tf.add(xent_loss, kl_loss, name='total_loss')
            tf.summary.scalar('xent_loss', xent_loss)
            tf.summary.scalar('kl_loss', kl_loss)
            tf.summary.scalar('vae_loss', self._losses[0])

    def _define_models(self):
        with tf.name_scope('data_input'):
            self._data = Input(shape=self._inputs_dims[0], name='data_input')
        with tf.name_scope('data_label'):
            self._label = Label(shape=self._inputs_dims[0])
        with tf.name_scope('mean_encoder'):
            x = self._data
            for dim in self._hiddens:
                x = Dense(dim, name='mean_encoder_inter', activation='elu')(x)
            self._latent_mean = Dense(
                self._latent_dim, name='mean_encoder')(x)
        with tf.name_scope('log_var_encoder'):
            x = self._data
            for dim in self._hiddens:
                x = Dense(dim, name='log_var_encoder_inter',
                          activation='elu')(x)
            self._latent_log_var = Dense(
                self._latent_dim, name='log_var_encoder')(x)
        with tf.name_scope('sample'):
            epsilon = tf.random_normal(
                shape=(self._batch_size, self._latent_dim), name='sampler')
            self._data_latent = self._latent_mean + \
                tf.exp(self._latent_log_var / 2) * epsilon

        with tf.name_scope('latent_input'):
            self._latent = Input((self._latent_dim,), name='latent_input')
        with tf.name_scope('decoder'):
            decoder_inter = []
            for dim in self._hiddens:
                decoder_inter.append(
                    Dense(dim, name='decoder_inter', activation='elu'))
            decoder_inter.append(
                Dense(self._inputs_dims[0][0], name='decoder', activation='sigmoid'))
            x_data = self._data_latent
            x_latent = self._latent
            for ly in decoder_inter:
                x_data = ly(x_data)
                x_latent = ly(x_latent)
            self._data_recon = x_data
            self._latent_recon = x_latent
        self._link_tensors()


class WGAN1D(NetGen):

    def __init__(self,
                 latent_dim=None,
                 settings=None,
                 **kwargs):
        NetGen.__init__(self, **kwargs)
        self._settings = settings
        self._models_names = ['WGan', 'Cri', 'Gen']
        self._is_train_step = [True, True, False]
        self._latent_dims = self._update_settings('latent_dim', latent_dim)

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

    def create_clip_step(self):
        self.clip_step = [w.assign(tf.clip_by_value(
            w, -0.01, 0.01)) for w in self._weights]

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
                    self._weights.append(ly.W)
                    x = ly(x)
            with tf.name_scope('recon'):
                ly = Dense(np.prod(self._inputs_dims))
                self._weights.append(ly.W)
                self._image = ly(x)

        lys = []
        with tf.name_scope('cri'):
            with tf.name_scope('reshape'):
                lys.append(Reshape([-1, 28, 28, 1]))
            with tf.name_scope('convs'):
                for dim in self._hiddens:
                    with tf.name_scope('features'):
                        lys.append(Convolution2D(dim, 3, 3, activation='elu'))
            with tf.name_scope('flatten'):
                lys.append(Flatten())
            with tf.name_scope('denses'):
                for dim in self._hiddens:
                    lys.append(Dense(dim, activation='elu'))
            with tf.name_scope('logits'):
                lys.append(Dense(1, activation='sigmoid'))

        for ly in lys:
            self._weights.append(ly.W)

        self.create_clip_step()

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

"""Base class for keras based models.
"""
# TODO: Test restore
# TODO: Add partial restore
# TODO: Add KAE

import logging
import numpy as np
import json
import tensorflow as tf
from keras.optimizers import Adam, RMSprop
from keras.objectives import mean_squared_error
from keras.models import Model, Sequential
from keras import backend as K

from ..utils.general import with_config, extend_list, zip_equal, empty_list
from ..utils.cells import Counter


class Net(object):
    """Base class for nets (hybrid of Keras and Tensorflow)
    This super class is designed to handle following common procedures of constructing a net:

    *   easy train/evaluation/prediction
    *   shortcuts for common parameters
    *   compile model with given loss, metrics, optimizer
    *   common callbacks
    *   save & load

    Users must _define_model(self) method. In which self._model is defined.

    Override of following methods is optional

    *   _define_loss(self)
    *   _define_metrics(self)
    *   _define_optimizer(self)

    All parameters are saved in settings
    Some shortcut to frequently used parameters are:

    *   _activ
    *   _arch
    *   _loss_name
    *   _metrics_name
    *   _optim_name
    *   _hiddens
    *   _is_dropout
    *   _is_save
    *   _is_load
    *   _lrs
    *   _path_load
    *   _path_save
    *   _hiddens
    *   _inputs_dim

    Parameter priority:
    command line(tf.flags) > python func args > file args(in order, later is higher) > default
    """

    @with_config
    def __init__(self,
                 inputs_dims=None,
                 outputs_dims=None,
                 batch_size=128,
                 optims_names=("RMSProp",),
                 losses_names=('mse',),
                 metrxs_names=(None,),
                 lrs=(1e-3,),
                 is_train_step=('True'),
                 is_save=(True,),
                 is_load=(False,),
                 path_saves=('./model.ckpt',),
                 path_loads=('./model.ckpt',),
                 path_summary=('./log',),
                 summary_freq=3,
                 arch='default',
                 activ='relu',
                 var_init='glorot_uniform',
                 hiddens=[],
                 is_dropout=False,
                 dropout_rate=0.5,
                 settings=None,
                 **kwargs):
        self._settings = settings
        if '_c' not in vars(self):
            self._c = dict()
        self._inputs_dims = self._update_settings(
            'inputs_dims', inputs_dims)
        self._outputs_dims = self._update_settings(
            'outputs_dims', outputs_dims)
        self._batch_size = self._update_settings(
            'batch_size', batch_size)
        self._summary_freq = self._update_settings(
            'summary_freq', summary_freq)

        self._optims_names = self._update_settings(
            'optims_names', optims_names)
        self._losses_names = self._update_settings(
            'losses_names', losses_names)
        self._metrxs_names = self._update_settings(
            'metrxs_names', metrxs_names)
        self._lrs = self._update_settings('lrs', lrs)
        self._lrs_tensor = None

        self._is_train_step = self._update_settings(
            'is_train_step', is_train_step)

        self._is_save = self._update_settings('is_save', is_save)
        self._is_load = self._update_settings('is_load', is_load)

        self._path_saves = self._update_settings(
            'path_saves', path_saves)
        self._path_loads = self._update_settings(
            'path_loads', path_loads)
        self._path_summary = self._update_settings(
            'path_summary', path_summary)

        self._arch = self._update_settings('arch', arch)
        self._activ = self._update_settings('activ', activ)
        self._var_init = self._update_settings('var_init', var_init)
        self._hiddens = self._update_settings('hiddens', hiddens)
        self._is_dropout = self._update_settings(
            'is_dropout', is_dropout)
        self._dropout_rate = self._update_settings(
            'dropout_rate', dropout_rate)
        self._filenames = self._update_settings('filenames', None)

        # Special variable, printable, but don't input by settings.
        self._models_names = self._update_settings(
            'model_names', ['Model'])
        self._nb_model = self._update_settings(
            'model_names', len(self._models_names))

        self._is_init = False

        if not self._is_init:
            self._initialize()

    def _initialize(self):
        self._inputs = None
        self._outputs = None
        self._labels = None
        self._models = None
        self._optims = None
        self._losses = None
        self._metrxs = None
        self._train_steps = None

        self._sess = tf.Session()
        self._step = Counter()
        self._summary_writer = None
        self._saver = None

        self._is_init = True

    def _update_settings(self, name, value=None):
        output = self._settings.get(name, value)
        self._c.update({name: output})
        return output

    def pretty_settings(self):
        """ print all settings in pretty JSON fashion """
        return json.dumps(self._c, sort_keys=True, indent=4, separators=(',', ':'))

    def _before_defines(self):
        # extent shareable parameters
        self._nb_model = len(self._models_names)
        self._models_names = extend_list(self._models_names, self._nb_model)
        self._optims_names = extend_list(self._optims_names, self._nb_model)
        self._losses_names = extend_list(self._losses_names, self._nb_model)
        self._metrxs_names = extend_list(self._metrxs_names, self._nb_model)
        self._lrs = extend_list(self._lrs, self._nb_model)
        self._path_summary = extend_list(self._path_summary, self._nb_model)
        self._path_saves = extend_list(self._path_saves, self._nb_model)
        self._path_loads = extend_list(self._path_loads, self._nb_model)
        self._is_train_step = extend_list(self._is_train_step, self._nb_model)
        self._is_load = extend_list(self._is_load, self._nb_model)

        self._inputs = empty_list(self._nb_model)
        self._outputs = empty_list(self._nb_model)
        self._labels = empty_list(self._nb_model)
        self._models = empty_list(self._nb_model)
        self._optims = empty_list(self._nb_model)
        self._losses = empty_list(self._nb_model)
        self._metrxs = empty_list(self._nb_model)
        self._train_steps = empty_list(self._nb_model)

        self._lrs_tensor = []
        for i in range(self._nb_model):
            self._lrs_tensor.append(tf.Variable(
                self._lrs[i], trainable=False, name='lr_%i' % i))

    def _define_models(self):
        """ define models """
        pass

    def _define_optims(self):
        """ define optimizers """
        for i in range(self._nb_model):
            opt_name = self._optims_names[i]
            lr_tensor = self._lrs_tensor[i]
            if opt_name == "Adam":
                self._optims[i] = tf.train.AdamOptimizer(
                    learning_rate=lr_tensor)
            elif opt_name == "RMSProp" or opt_name == "rmsporp" or opt_name == "rms":
                self._optims[i] = tf.train.RMSPropOptimizer(
                    learning_rate=lr_tensor)
            else:
                raise ValueError("Unknown optimizer name %s." % opt_name)

    def _define_losses(self):
        for i in range(self._nb_model):
            if not self._is_train_step[i]:
                continue
            loss_name = self._losses_names[i]
            if loss_name == 'mse':
                self._losses[i] = tf.losses.mean_squared_error(
                    self._labels[i][0], self._outputs[i][0])
            loss_summary_name = self._models_names[i] + '_loss'
            tf.summary.scalar(name=loss_summary_name, tensor=self._losses[i])

    def _define_metrxs(self):
        # TODO: Impl
        pass

    def _define_train_steps(self):
        self._train_steps = empty_list(self._nb_model)
        for i in range(self._nb_model):
            if self._is_train_step[i]:
                self._train_steps[i] = self._optims[
                    i].minimize(self._losses[i])
            else:
                self._train_steps[i] = None

    def _define_summaries(self):
        self._sess.run(tf.global_variables_initializer())
        self._summary_writer = tf.summary.FileWriter(
            self._path_summary[0], self._sess.graph)
        self._summaries = tf.summary.merge_all()
        self._saver = tf.train.Saver()

    def load(self, var_names=None, model_id=None):
        """ load weight from file """
        # TODO: Impl: restore from checkpoint files
        if model_id is None:
            model_id = 0
        # with open(self._path_loads[model_id]) as fin:
        #     for info in fin.readlines():
        #         infos = info.split(" ")
        #         if infos[0] == 'all_model_checkpoint_paths:':
        #             path_restore = infos[1]
        self._saver.restore(self._sess, self._path_loads[model_id])

    def save(self, var_names=None, model_id=None, is_print=False):
        # TODO: Impl partial save.
        if model_id is None:
            model_id = 0
        # self._saver.save(self._sess, self._path_saves[
        #                  model_id], global_step=self._step.state)
        path = self._saver.save(self._sess, self._path_saves[model_id])
        if is_print:
            print('Net saved in:')
            print(path)

    def model_id(self, name):
        return self._models_names.index(name)

    def define_net(self):
        """ Compile the model"""
        self._before_defines()
        self._define_models()
        self._define_losses()
        self._define_metrxs()
        self._define_optims()
        self._define_train_steps()
        self._define_summaries()
        for i in range(self._nb_model):
            if self._is_load[i]:
                self.load(model_id=i)

    def train_on_batch(self, model_id=None, inputs=None, labels=None, is_summary=None):
        """ train on a batch of data
        Args:
            model_id: Id of model to train, maybe fetched by model_id(model_name). Default = 0.
            inputs: *List* of input tensors
            labels: *List* of label tensors
            is_summary: flag of dumping summary, leave it to None for auto summary with summay_freq.
        Returns:
            loss_v: value of loss of current step.
        """
        if inputs is None:
            inputs = []
        if labels is None:
            labels = []
        if model_id is None:
            model_id = 0
        next(self._step)
        if isinstance(model_id, str):
            model_id = self.model_id(model_id)
        feed_dict = {}
        ip_tensors = self._inputs[model_id]
        lb_tensors = self._labels[model_id]
        if ip_tensors is None:
            ip_tensors = []
        if lb_tensors is None:
            lb_tensors = []
        for (tensor, data) in zip_equal(ip_tensors, inputs):
            feed_dict.update({tensor: data})
        for (tensor, data) in zip_equal(lb_tensors, labels):
            feed_dict.update({tensor: data})
        feed_dict.update({K.learning_phase(): 1})
        for (tensor, value) in zip_equal(self._lrs_tensor, self._lrs):
            feed_dict.update({tensor: value})
        if is_summary is None:
            is_summary = (self._step.state % self._summary_freq == 0)
        train_step = self._train_steps[model_id]
        if not is_summary:
            _, loss_v = self._sess.run(
                [train_step, self._losses[model_id]], feed_dict=feed_dict)
        else:
            _, loss_v, summary_v = self._sess.run(
                [train_step, self._losses[model_id], self._summaries], feed_dict=feed_dict)
            self._summary_writer.add_summary(summary_v, self._step.state)
        return loss_v

    def predict(self, model_id=None, inputs=None):
        if model_id is None:
            model_id = 0
        if isinstance(model_id, str):
            model_id = self.model_id(model_id)
        feed_dict = {}
        ip_tensors = self._inputs[model_id]
        for (tensor, data) in zip_equal(ip_tensors, inputs):
            feed_dict.update({tensor: data})
        feed_dict.update({K.learning_phase(): 0})
        predicts = self._sess.run(self._outputs[model_id], feed_dict=feed_dict)
        return predicts

    def reset_lr(self, lrs):
        self._lrs = extend_list(lrs, self._nb_model)

    def lr_decay(self):
        lr_v = self._lrs[0]
        self._lrs = extend_list([lr_v], self._nb_model)

    # def plot_loss(self, model_id=0, sub_id=None, is_clean=True, is_log=False, smooth=0.0):
    #     if is_clean:
    #         display.clear_output(wait=True)
    #         display.display(plt.gcf())
    #     if sub_id is None:
    #         l = self._loss_records[model_id]
    #     else:
    #         l = self._loss_records[model_id][sub_id]
    #     l_s = np.array(l)
    #     l_s[0] = l[0]
    #     for i in range(len(l) - 1):
    #         l_s[i + 1] = l_s[i] * smooth + l[i + 1]
    #     l = l_s
    #     if is_log:
    #         l = np.log(l)
    #     plt.plot(l)
    #     return plt

    # def plot_model(self, model_id=0, is_IPython=True, filename=None, show_shapes=True):
    #     if isinstance(model_id, str):
    #         model_id = self.get_model_id(model_id)
    #     if is_IPython:
    #         display.SVG(kvu.model_to_dot(self._models[model_id], show_shapes=show_shapes).create(
    #             prog='dot', format='svg'))
    #     else:
    #         kvu.plot(model, show_shapes=show_shapes, to_file='model.png')

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def sess(self):
        return self._sess

    @property
    def summary_writer(self):
        return self._summary_writer


class NetGen(Net):

    @with_config
    def __init__(self,
                 latent_dis='gaussian',
                 latent_dims=2,
                 latent_MoG_mode=10,
                 latent_sigma=1.0,
                 latent_MoG_mus=None,
                 settings=None,
                 **kwargs):
        super(NetGen, self).__init__(**kwargs)
        self._settings = settings
        self._latent_dis = self._update_settings(
            'latent_dis', latent_dis)
        self._latent_dims = self._update_settings(
            'latent_dims', latent_dims)
        self._latent_MoG_mode = self._update_settings(
            'latent_MoG_mode', latent_MoG_mode)
        self._latent_MoG_mus = self._update_settings(
            'latent_MoG_mus', latent_MoG_mus)
        self._latent_sigma = self._update_settings(
            'latent_sigma', latent_sigma)

    def gen_latent(self):
        if self._latent_dis == 'gaussian':
            return np.random.randn(self._batch_size, self._latent_dims) * self._latent_sigma
        elif self._latent_dis == 'uniform':
            return np.random.rand(self._batch_size, self._latent_dims) * self._latent_sigma - self._latent_sigma / 2
        elif self._latent_dis == 'MoG':
            id_gaussian = np.random.randint(
                0, self._latent_MoG_mode)
            value = np.random.randn(
                self._batch_size, self._latent_dims) * self._latent_sigma
            value += self._latent_MoG_mus[id_gaussian]
            return value

    def gen_data(self, latent=None):
        if latent is None:
            return self.predict(self.model_id('Gen'), self.gen_latent())
        else:
            return self.predict(self.model_id('Gen'), latent)

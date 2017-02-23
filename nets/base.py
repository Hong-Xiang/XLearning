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
                 inputs_dims=[],
                 outputs_dims=[],
                 settings=None,
                 **kwargs):

        self._inputs_dims = settings.get('inputs_dims', inputs_dims)
        self._outputs_dims = settings.get('outputs_dims', outputs_dims)
        self._batch_size = settings.get('batch_size', 128)
        self._summary_freq = settings.get('summary_freq', 10)

        self._optims_names = settings.get('optims_names', ['Adam'])
        self._losses_names = settings.get('losses_names', ['mse'])
        self._metrxs_names = settings.get('metrxs_names', [None])
        self._lrs = settings.get('lrs', [1e-4])

        self._is_train_step = settings.get('is_train_step', [True])

        self._is_save = settings.get('is_save', [True])
        self._is_load = settings.get('is_load', [False])

        self._path_saves = settings.get('path_saves', ['./model.ckpt'])
        self._path_loads = settings.get('path_loads', ['./model.ckpt'])
        self._path_summary = settings.get('path_summary', ['./log'])

        self._arch = settings.get('arch', 'default')
        self._activ = settings.get('activ', 'relu')
        self._var_init = settings.get('var_init', 'glorot_uniform')
        self._hiddens = settings.get('hiddens', [])
        self._is_dropout = settings.get('is_dropout', False)
        self._dropout_rate = settings.get('dropout_rate', 0.5)

        self._c = dict()
        self._c.update({'inputs_dims': self._inputs_dims})
        self._c.update({'outputs_dims': self._outputs_dims})
        self._c.update({'hiddens': self._hiddens})
        self._c.update({'batch_size': self._batch_size})
        self._c.update({'var_init': self._var_init})
        self._c.update({'optims_names': self._optims_names})
        self._c.update({'losses_names': self._losses_names})
        self._c.update({'metrxs_names': self._metrxs_names})
        self._c.update({'lrs': self._lrs})
        self._c.update({'is_save': self._is_save})
        self._c.update({'is_load': self._is_load})
        self._c.update({'path_saves': self._path_saves})
        self._c.update({'path_loads': self._path_loads})
        self._c.update({'arch': self._arch})
        self._c.update({'var_init': self._var_init})
        self._c.update({'hiddens': self._hiddens})
        self._c.update({'is_dropout': self._is_dropout})
        self._c.update({'dropout_rate': self._dropout_rate})

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

        self._models_names = ['Model']
        self._nb_model = None

    def print_settings(self):
        print(json.dumps(self._c, sort_keys=True, separators=(':', ','), indent=4))

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

        self._inputs = empty_list(self._nb_model)
        self._outputs = empty_list(self._nb_model)
        self._labels = empty_list(self._nb_model)
        self._models = empty_list(self._nb_model)
        self._optims = empty_list(self._nb_model)
        self._losses = empty_list(self._nb_model)
        self._metrxs = empty_list(self._nb_model)
        self._train_steps = empty_list(self._nb_model)

    def _define_models(self):
        """ define models """
        pass

    def _define_optims(self):
        """ define optimizers """
        for i in range(self._nb_model):
            opt_name = self._optims_names[i]
            lr = self._lrs[i]
            if opt_name == "Adam":
                self._optims[i] = tf.train.AdamOptimizer(learning_rate=lr)
            elif opt_name == "RMSProp" or opt_name == "rmsporp" or opt_name == "rms":
                self._optims[i] = tf.train.RMSPropOptimizer(learning_rate=lr)
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
            tf.summary.scalar(name=self._models_names[
                              i] + '_loss', tensor=self._losses[i])

    def _define_metrxs(self):
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
        if model_id is None:
            model_id = 0
        # with open(self._path_loads[model_id]) as fin:
        #     for info in fin.readlines():
        #         infos = info.split(" ")
        #         if infos[0] == 'all_model_checkpoint_paths:':
        #             path_restore = infos[1]
        self._saver.restore(self._sess, self._path_loads[model_id])

    def save(self, var_names=None, model_id=None):
        # TODO: Impl partial save.
        if model_id is None:
            model_id = 0
        # self._saver.save(self._sess, self._path_saves[
        #                  model_id], global_step=self._step.state)
        self._saver.save(self._sess, self._path_saves[model_id])

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

        # self.load()

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
        if model_id is None:
            model_id = 0
        next(self._step)
        if isinstance(model_id, str):
            model_id = self.model_id(model_id)
        feed_dict = {}
        ip_tensors = self._inputs[model_id]
        lb_tensors = self._labels[model_id]
        for (tensor, data) in zip_equal(ip_tensors, inputs):
            feed_dict.update({tensor: data})
        for (tensor, data) in zip_equal(lb_tensors, labels):
            feed_dict.update({tensor: data})
        feed_dict.update({K.learning_phase(): 1})
        if is_summary is None:
            is_summary = (self._step.state % self._summary_freq == 0)
        train_step = self._train_steps[model_id]
        if is_summary:
            loss_v = self._sess.run(train_step, feed_dict=feed_dict)
        else:
            loss_v, summary_v = self._sess.run(
                [train_step, self._summaries], feed_dict=feed_dict)
            self._summary_writer.add_summary(summary_v, self._step.state)
        return loss_v

    def predict(self, model_id=None, inputs=None, **kwargs):
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
        self._define_optims()
        self._define_train_steps()

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


# class KGen(KNet):

#     def __init__(self, **kwargs):
#         super(KGen, self).__init__(**kwargs)
#         self._model_gen = None

#     def gen_noise(self):
#         return np.random.randn(self._batch_size, self._encoding_dim)
#         # return np.random.normal(size=(self._batch_size, self._encoding_dim))

#     def gen_data(self, noise=None):
#         if noise is None:
#             return self._model_gen.predict(self.gen_noise())
#         else:
#             return self._model_gen.predict(noise)

#     @property
#     def model_gen(self):
#         return self._model_gen

"""Base class for keras based models.
"""
# TODO: Add restore
# TODO: Add partial restore
# TODO: Add convinient fit, predict, evaluate (given dataset object)
# TODO: Modify KGen into decorator
# TODO: Add KAE
# TODO: Debug plot_loss

import logging
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Model, Sequential
from keras import backend as K
import keras.utils.visualize_util as kvu

from ..utils.general import with_config


class KNet(object):
    """Base class for keras nets.
    This super class is designed to handle following common procedures of constructing a net:

    *   common parameters
    *   compile model with given loss, metrics, optimizer
    *   common callbacks
    *   save & load

    Users must _define_model(self) method. In which self._model is defined.

    Override of following methods is optional

    *   _define_loss(self)
    *   _define_metrics(self)
    *   _define_optimizer(self)

    All parameters are saved in self.settings
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
    *   _lr
    *   _path_load
    """

    @with_config
    def __init__(self, settings=None, **kwargs):
        self._settings = settings

        # models

        self._models = []
        self._optims = []
        self._losses = []
        self._metrxs = []
        self._loss_records = []

        self._callbacks = []

        # settings
        # general:

        # default activ
        self._activ = self._settings.get('activ', 'relu')

        # a list convinient input of # of hidden units
        self._hiddens = self._settings.get('hiddens', [])
        self._batch_size = self._settings.get('batch_size', 128)
        self._init = self._settings.get('init', 'glorot_uniform')

        self._is_dropout = self._settings.get('is_dropout', False)
        self._dropout_rate = self._settings.get('dropout_rate', 0.5)

        self._optims_names = self._settings.get('optims_names', ['Adam'])
        self._losses_names = self._settings.get('losses_names', ['mse'])
        self._metrxs_names = self._settings.get('metrxs_names', [None])
        self._lrs = self._settings.get('lrs', [1e-4])

        # s/l:
        self._is_save = self._settings.get('is_save', [True])
        self._is_load = self._settings.get('is_load', [False])
        self._path_saves = self._settings.get('path_saves', ['./save'])
        self._path_loads = self._settings.get('path_loads', ['./save'])

        self._arch = self._settings.get('arch', 'default')

        self._input_dim = self._settings.get('input_dim', (128, ))
        self._encoding_dim = self._settings.get('encoding_dim', 128)

        self._nb_model = 1

    def _standarize(self):
        if len(self._optims_names) == 1 and self._nb_model > 1:
            self._optims_names *= self._nb_model
        if len(self._losses_names) == 1 and self._nb_model > 1:
            self._losses_names *= self._nb_model
        if len(self._metrxs_names) == 1 and self._nb_model > 1:
            self._metrxs_names *= self._nb_model
        if len(self._lrs) == 1 and self._nb_model > 1:
            self._lrs *= self._nb_model

        for i in range(self._nb_model):
            self._models.append([])
            self._optims.append([])
            self._losses.append([])
            self._metrxs.append([])
            self._loss_records.append([])

    def _define_models(self):
        """ define model """
        pass

    def _define_optims(self):
        """ optimizer """
        self._optims = []
        for (opt_name, lr) in zip(self._optims_names, self._lrs):
            if opt_name == "Adam":
                self._optims.append(Adam(lr))
            elif opt_name == "RMSProp" or opt_name == "rmsporp" or opt_name == "rms":
                self._optims.append(RMSprop(lr=lr))
            else:
                raise ValueError("Unknown optimizer name %s." % opt_name)

    def _define_losses(self):
        self._losses = self._losses_names

    def _define_metrxs(self):
        self._metrxs = self._metrxs_names

    def _compile_models(self):
        logging.getLogger(__name__).debug(
            'len(models): %d.' % len(self._models))
        logging.getLogger(__name__).debug(
            'len(optims): %d.' % len(self._optims))
        logging.getLogger(__name__).debug(
            'len(losses): %d.' % len(self._losses))
        logging.getLogger(__name__).debug(
            'len(metrxs): %d.' % len(self._metrxs))
        for md, opt, loss, metric in zip(self._models, self._optims, self._losses, self._metrxs):
            md.compile(optimizer=opt, loss=loss, metrics=metric)
            md.summary()

    def _define_callbks(self):
        for pathsave in self._path_saves:
            tmpbk = []
            ckp = ModelCheckpoint(pathsave, monitor='loss', verbose=0,
                                  save_best_only=False, save_weights_only=False, mode='auto', period=10)
            tsb = TensorBoard(write_graph=False, write_images=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                          patience=10, min_lr=1e-8)
            tmpbk.append(tsb)
            tmpbk.append(reduce_lr)
            tmpbk.append(ckp)
            self._callbacks.append(tmpbk)

    def load_weights(self):
        """ load weight from file """
        for md, filepath, flag in zip(self._models, self._path_loads, self._is_load):
            if flag:
                md.load_weights(filepath, by_name=True)

    def get_model_id(self, name):
        return 0

    def define_net(self):
        """ Compile the model"""
        self._standarize()
        self._define_losses()
        self._define_metrxs()
        self._define_optims()
        self._define_models()
        self._compile_models()
        self._define_callbks()
        self.load_weights()

    def train_on_batch(self, model_id, inputs, outputs, **kwargs):
        if isinstance(model_id, str):
            model_id = self.get_model_id(model_id)

        loss_now = self._models[model_id].train_on_batch(
            inputs, outputs, **kwargs)
        self._loss_records[model_id].append(loss_now)

    def reset_lr(self, lrs):
        if len(lrs) == 1 and len(self._optims) > 1:
            lrs = lrs * len(self._optims)
        for (opt, lrv) in zip(self._optims, lrs):
            K.set_value(opt.lr, lrv)

    def plot_loss(self, model_id=0, sub_id=None, is_clean=True, is_log=False, smooth=0.0):
        if is_clean:
            display.clear_output(wait=True)
            display.display(plt.gcf())
        if sub_id is None:
            l = self._loss_records[model_id]
        else:
            l = self._loss_records[model_id][sub_id]
        l_s = np.array(l)
        l_s[0] = l[0]
        for i in range(len(l)-1):
            l_s[i+1] = l_s[i] * smooth + l[i+1]
        l = l_s
        if is_log:
            l = np.log(l)        
        plt.plot(l)
        return plt

    def plot_model(self, model_id=0, is_IPython=True, filename=None, show_shapes=True):
        if isinstance(model_id, str):
            model_id = self.get_model_id(model_id)
        if is_IPython:
            display.SVG(kvu.model_to_dot(self._models[model_id], show_shapes=show_shapes).create(
                prog='dot', format='svg'))
        else:
            kvu.plot(model, show_shapes=show_shapes, to_file='model.png')

    @property
    def model(self):
        """ model """
        return self._models

    @property
    def callbacks(self):
        """ call back list"""
        return self._callbacks

    @property
    def optims(self):
        return self._optims

    @property
    def loss_records(self):
        return self._loss_records

    @property
    def batch_size(self):
        return self._batch_size


class KGen(KNet):

    def __init__(self, **kwargs):
        super(KGen, self).__init__(**kwargs)
        self._model_gen = None

    def gen_noise(self):
        return np.random.randn(self._batch_size, self._encoding_dim)
        # return np.random.normal(size=(self._batch_size, self._encoding_dim))

    def gen_data(self):
        return self._model_gen.predict(self.gen_noise())

    @property
    def model_gen(self):
        return self._model_gen

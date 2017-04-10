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
import json
import matplotlib.pyplot as plt
from IPython import display
import tensorflow as tf

from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Model, Sequential
# from keras.losses import mean_square_error
from keras import backend as K
import keras.utils.vis_utils as kvu
import time

from ..utils.general import with_config, extend_list, zip_equal, empty_list, get_args


class Net(object):
    """Base class for nets (hybrid of Keras and Tensorflow)
    This super class is designed to handle following common procedures of constructing a net:

    *   easy train/evaluation/prediction
    *   common parameters fields and general parameter handling
    *   compile model with given loss, metrics, optimizer
    *   common callbacks
    *   save & load

    Users must _define_model(self) method. In which self._model is defined.

    Override of following methods is optional

    *   _define_loss(self)
    *   _define_metrics(self)
    *   _define_optimizer(self)

    All parameters are saved in self._c
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
    *   _inputs_shapes
    *   _outputs_shapes

    Parameter priority:
    command line(tf.flags) > python func args > file args(in order, later is higher) > default
    """

    @with_config
    def __init__(self,
                 inputs_shapes=None,
                 outputs_shapes=None,
                 batch_size=None,
                 optims_names=("RMSProp",),
                 losses_names=('mse',),
                 metrxs_names=(None,),
                 lrs=(1e-3,),
                 is_trainable=(True,),
                 is_save=(True,),
                 is_load=(False,),
                 is_bn=True,
                 path_save='save',
                 save_freq=100,
                 path_load='save',
                 path_summary=('./log',),
                 summary_freq=100,
                 arch='default',
                 activ='elu',
                 var_init='glorot_gaussian',
                 hiddens=None,
                 is_dropout=False,
                 dropout_rate=0.5,
                 init_step=0,
                 filenames=[],
                 **kwargs):
        self._inputs_shapes = inputs_shapes
        self._outputs_shapes = outputs_shapes
        self._batch_size = batch_size
        self._summary_freq = summary_freq

        self._optims_names = optims_names
        self._losses_names = losses_names
        self._metrxs_names = metrxs_names
        self._lrs = lrs

        self._is_trainable = is_trainable

        self._is_save = is_save
        self._is_load = is_load

        self._path_save = path_save
        self._save_freq = save_freq
        self._path_load = path_load
        self._path_summary = path_summary

        self._arch = arch
        self._activ = activ
        self._var_init = var_init
        self._hiddens = hiddens
        self._is_dropout = is_dropout
        self._dropout_rate = dropout_rate
        self._is_bn = is_bn

        self._init_step = init_step
        self._filenames = filenames

        # Special variable, printable, but don't input by settings.
        self._models_names = ['Model']
        self._nb_model = len(self._models_names)

        self._callbacks = []
        self._is_init = False

        self._loss_records = []

        self.global_step = 0

        self._skipkeys = ['_c', '_skipkeys', '_models', '_optims',
                          '_losses', '_labels', '_outputs', '_loss_records', '_metrxs', '_inputs', '_callbacks']

        self._last_save = None

    def _initialize(self):
        pass

    def _update_settings(self):
        self._c = dict()

        for v in vars(self):
            if v[0] != "_":
                continue
            if v[1] == "_":
                continue
            if v in self._skipkeys:
                continue
            self._c.update({v[1:]: getattr(self, v)})

    def pretty_settings(self):
        """ print all settings in pretty JSON fashion """
        LEN = 50
        prefix = "=" * LEN + "\n"
        prefix += str(self.__class__) + " settings:" + "\n"
        prefix += "." * LEN + "\n"
        sets = json.dumps(self._c, sort_keys=True,
                          indent=4, separators=(',', ': '))
        suffix = "\n" + "=" * LEN
        return prefix + sets + suffix

    def _before_defines(self):
        # extent shareable parameters
        self._nb_model = len(self._models_names)
        self._models_names = extend_list(self._models_names, self._nb_model)
        self._optims_names = extend_list(self._optims_names, self._nb_model)
        self._losses_names = extend_list(self._losses_names, self._nb_model)
        self._metrxs_names = extend_list(self._metrxs_names, self._nb_model)
        self._lrs = extend_list(self._lrs, self._nb_model)
        self._path_summary = extend_list(self._path_summary, self._nb_model)
        self._is_trainable = extend_list(self._is_trainable, self._nb_model)
        self._is_load = extend_list(self._is_load, self._nb_model)
        self._is_save = extend_list(self._is_save, self._nb_model)

        self._inputs = empty_list(self._nb_model)
        self._outputs = empty_list(self._nb_model)
        self._labels = empty_list(self._nb_model)
        self._models = empty_list(self._nb_model)
        self._optims = empty_list(self._nb_model)
        self._losses = empty_list(self._nb_model)
        self._metrxs = empty_list(self._nb_model)

        for i in range(self._nb_model):
            self._loss_records.append({})

        self._update_settings()

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
        for md, opt, loss, metric in zip(self._models, self._optims, self._losses, self._metrxs):
            md.compile(optimizer=opt, loss=loss, metrics=metric)
            # md.summary()

    def _define_callbks(self):
        # for pathsave in self._path_saves:
        #     tmpbk = []
        #     ckp = ModelCheckpoint(pathsave, monitor='loss', verbose=0,
        #                           save_best_only=False, save_weights_only=False, mode='auto', period=10)
        #     tsb = TensorBoard(write_graph=True, write_images=True)
        #     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
        #                                   patience=10, min_lr=1e-8)
        #     tmpbk.append(tsb)
        #     tmpbk.append(reduce_lr)
        #     tmpbk.append(ckp)
        #     self._callbacks.append(tmpbk
        pass

    def save(self, model_id=None, file_path=None, step='ukn'):
        if model_id is None:
            for i in range(self._nb_model):
                if self._is_save[i]:
                    self.save(self._models_names[i],
                              file_path=file_path, step=step)
        else:
            if isinstance(model_id, str):
                model_id = self._models_names.index(model_id)
            model_name = self._models_names[model_id]
            if file_path is None:
                file_path = "{0}-{1}-{2}".format(self._path_save,
                                                 model_name, step)
            print("Saving model: {0:10} to {1:10}.".format(
                model_name, file_path))
            self.model(model_id).save_weights(file_path)

    def load(self, model_id=None, file_path=None, step=None, is_force=False):
        """ load weight from file """
        if model_id is None:            
            for i in range(self._nb_model):
                if self._is_load[i] or is_force:
                    self.load(
                        self._models_names[i], file_path=file_path, step=step, is_force=is_force)
        else:            
            if step is None:
                return
            if file_path is None:
                if isinstance(model_id, str):
                    model_id = self._models_names.index(model_id)
                model_name = self._models_names[model_id]
                file_path = "{0}-{1}-{2}".format(self._path_save,
                                                 model_name, step)
                print("Loading model: {0:10} from {1:10}.".format(
                    model_name, file_path))
                self.model(model_id).load_weights(file_path, by_name=True)

    def dump_loss(self, filename='loss.npy'):
        loss_t = np.zeros(shape=(self.global_step + 10, self._nb_model))
        eps = 1e-7
        for i in range(self._nb_model):
            for k, v in self._loss_records[i].items():
                loss_t[k, i] = v
        for i in range(self._nb_model):
            for j in range(self.global_step - 1):
                if abs(loss_t[j + 1, i]) < eps:
                    loss_t[j + 1, i] = loss_t[j, i]
        np.save(filename, loss_t)

    def define_net(self):
        """ Compile the model"""
        self._before_defines()
        self._define_losses()
        self._define_metrxs()
        self._define_optims()
        self._define_models()
        self._compile_models()
        self._define_callbks()
        self.load()

    def _train_model_on_batch(self, model_id, inputs, outputs):
        m = self.model(model_id)
        loss_v = m.train_on_batch(inputs, outputs)
        return loss_v
    
    def _scadule_model(self):
        return self._models_names[0]

    def train_on_batch(self, model_id=None, inputs=None, outputs=None, **kwargs):
        if model_id is None:
            model_id = self._scadule_model()
        loss_now = self._train_model_on_batch(model_id, inputs, outputs)
        self.global_step += 1
        if isinstance(model_id, str):
            model_id = self._models_names.index(model_id)
        self._loss_records[model_id].update({self.global_step: loss_now})
        # self._loss_records[model_id].append(loss_now)
        save_flag = False
        if self._save_freq > 0:
            if self.global_step % self._save_freq == 0:
                save_flag = True
        if self._last_save is None:
            self._last_save = time.time()
        else:
            delta = time.time() - self._last_save
            if delta > 30.0 * 60.0:
                save_flag = True
        if save_flag:
            self.save(step=self.global_step)
            self._last_save = time.time()
        if self._summary_freq > 0:
            if self.global_step % self._summary_freq == 0:
                self.dump_loss()
        return loss_now

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
        for i in range(len(l) - 1):
            l_s[i + 1] = l_s[i] * smooth + l[i + 1]
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

    def model(self, id_or_name):
        """ Get model ref by id or model name """
        return self._models[self.model_id(id_or_name)]

    def model_id(self, id_or_name):
        if isinstance(id_or_name, str):
            m_id = self._models_names.index(id_or_name)
        else:
            m_id = int(id_or_name)
        return m_id

    @property
    def callbacks(self):
        """ call back list"""
        return self._callbacks

    @property
    def optims(self):
        return self._optims

    @property
    def batch_size(self):
        return self._batch_size

    def fit_full(self, data_generator):
        raise NotImplementedError('No fit_full implementation.')

    def _predict(self, model_id, inputs):
        return self.model(model_id).predict(inputs, batch_size=self._batch_size)

    def predict(self, model_id, inputs):
        return self._predict(model_id, inputs)


class KAE(Net):
    @with_config
    def __init__(self,
                 latent_dim=None,
                 settings=None,
                 **kwargs):
        Net.__init__(self, **kwargs)
        self._settings = settings
        self._latent_dim = self._update_settings('latent_dim', latent_dim)

    @property
    def model_ae(self):
        return self.model('ae')

    @property
    def model_enc(self):
        return self.model('enc')

    @property
    def model_dec(self):
        return self.model('dec')


class KGen(Net):
    @with_config
    def __init__(self,
                 latent_dim=None,
                 settings=None,
                 **kwargs):
        Net.__init__(self, **kwargs)
        self._settings = settings
        self._latent_dim = self._update_settings('latent_dim', latent_dim)

    def gen_noise(self):
        return np.random.randn(self._batch_size, self._latent_dim)
        # return np.random.normal(size=(self._batch_size, self._encoding_dim))

    def gen_data(self):
        return self._model_gen.predict(self.gen_noise())

    @property
    def model_gen(self):
        return self.model('gen')

"""Base class for keras based models.
"""
# TODO: Add restore
# TODO: Add partial restore

import xlearn.utils.general as utg
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Model, Sequential
from keras import backend as K


class KNet(object):
    """Base class for keras nets.
    """

    def __init__(self, filenames=None, **kwargs):
        self._settings = utg.merge_settings(
            settings=None, filenames=filenames, **kwargs)

        # model
        self._model = None
        self._optim = None
        self._callbacks = []

        # settings
        # general:
        self._activation = self._settings.get('activation', 'relu')
        self._hiddens = self._settings.get('hiddens', [])
        self._is_dropout = self._settings.get('is_dropout', False)
        self._batch_size = self._settings.get('batch_size', 128)
        self._architecture = self._settings.get('architescture', 'default')
        self._optimizer_name = self._settings.get('optimizer', 'Adam')

        # learning rate:
        self._lr = self._settings['lr_init']

        # loss:
        self._loss = self._settings['loss']
        self._metrics = self._settings.get('metrics', None)

    def _define_model(self):
        """ define model """
        pass

    def _define_optimizer(self):
        """ optimizer """
        if self._optimizer_name == "Adam":
            self._optim = Adam(self._lr)

    def load_weights(self, filepath):
        """ load weight from file """
        self._model.load_weights(filepath, by_name=True)

    def define_net(self):
        """ Compile the model"""
        self._define_model()
        self._define_optimizer()
        self._model.compile(optimizer=self._optim,
                            loss=self._loss, metrics=self._metrics)
        self._model.summary()
        ckp = ModelCheckpoint("./save", monitor='loss', verbose=0,
                              save_best_only=False, save_weights_only=False, mode='auto', period=10)
        tsb = TensorBoard(write_graph=False, write_images=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                      patience=10, min_lr=1e-8)
        self._callbacks.append(tsb)
        self._callbacks.append(reduce_lr)
        self._callbacks.append(ckp)

    @property
    def model(self):
        """ model """
        return self._model

    @property
    def callbacks(self):
        """ call back list"""
        return self._callbacks

    @property
    def lr(self):
        """ initial learning rate """
        return K.get_value(self.model.optimizer.lr)

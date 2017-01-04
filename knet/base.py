"""Base class for keras based models.
"""

import xlearn.utils.general as utg
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


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
        self.batch_size = self._settings.get('batch_size', 128)

        # learning rate:
        self._lr = self._settings['lr_init']
        self._lr_is_decay = self._settings.get('lr_is_decay', False)

        # loss:
        self._loss = self._settings['loss']
        self._metrics = self._settings['metrics']

    def _define_model(self):
        """ define model """
        pass

    def _define_optimizer(self):
        """ optimizer """
        pass

    def define_net(self):
        """ Compile the model"""
        self._define_model()
        self._define_optimizer()
        self._model.compile(optimizer=self._optim,
                            loss=self._loss, metrics=self._metrics)
        self._model.summary()
        tsb = TensorBoard(write_graph=True, write_images=True)
        self._callbacks.append(tsb)

    @property
    def model(self):
        """ model """
        return self._model

    @property
    def callbacks(self):
        """ call back list"""
        return self._callbacks

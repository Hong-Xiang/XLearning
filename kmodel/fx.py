"""Deep net for fx based on keras.
"""
from xlearn.kmodel.base import KNet
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import ELU
from keras.optimizers import Nadam, Adam


class KNetFx(KNet):

    def __init__(self, filenames=None, **kwargs):
        super(KNetFx, self).__init__(filenames=filenames, **kwargs)

    def _gather_settings(self):
        super(KNetFx, self)._gather_settings()
        self._hidden_units = self._settings['hidden_units_list']
        self._is_dropout = self._settings['is_dropout']
        self._lr = self._settings['lr_init']

    def _net_definition(self):
        model = Sequential()
        n_hidden_layers = len(self._hidden_units)
        model.add(Dense(self.n_hidden[0], input_dim=(1)))
        for i in range(1, n_hidden_layers):
            # model.add(Activation('relu'))
            model.add(ELU())
            if self._is_dropout:
                model.add(Dropout(0.5))
            model.add(Dense(self.n_hidden[i]))
        # model.add(Activation('relu'))
        model.add(ELU())
        if self._is_dropout:
            model.add(Dropout(0.5))
        model.add(Dense(1))
        nadam = Nadam(lr=self._lr)
        adam = Adam(lr=self._lr, decay=0.0001)
        model.compile(optimizer=adam, loss='mse',
                      metrics=['mean_squared_error'])
        return model

    @property
    def n_hidden(self):
        return self._hidden_units


class KNetFx2(KNet):

    def __init__(self, filenames=None, **kwargs):
        super(KNetFx2, self).__init__(filenames=filenames, **kwargs)

    def _gather_settings(self):
        super(KNetFx2, self)._gather_settings()
        self._hidden_units = self._settings['hidden_units_list']
        self._is_dropout = self._settings['is_dropout']
        self._lr = self._settings['lr_init']

    def _net_definition(self):
        model = Sequential()
        n_hidden_layers = len(self._hidden_units)
        model.add(Dense(self.n_hidden[0], input_dim=(1)))
        for i in range(1, n_hidden_layers):            
            model.add(ELU())
            if self._is_dropout:
                model.add(Dropout(0.5))
            model.add(Dense(self.n_hidden[i]))        
        model.add(ELU())
        if self._is_dropout:
            model.add(Dropout(0.5))
        model.add(Dense(1))
        nadam = Nadam(lr=self._lr)
        adam = Adam(lr=self._lr, decay=0.0001)
        model.compile(optimizer=adam, loss='mse',
                      metrics=['mean_squared_error'])
        return model

    @property
    def n_hidden(self):
        return self._hidden_units
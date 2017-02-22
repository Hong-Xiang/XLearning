from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import objectives
import keras.backend as K
import numpy as np

from .base import KNet
from ..kmodel.dense import denses


class MLP(KNet):

    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)

    def _define_models(self):
        ip = Input(shape=self._input_dim)
        h = denses(ip, hiddens=self._hiddens,
                   is_dropout=self._is_dropout, dropout_rate=self._dropout_rate)
        y = Dense(2)(h)
        self._models[0] = Model(input=ip, output=y)




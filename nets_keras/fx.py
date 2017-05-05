"""Deep net for fx based on keras.
"""
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Nadam, Adam
from xlearn.knet.base import KNet
from ..kmodel.dense import denses

class KNetFx(KNet):

    def __init__(self, filenames=None, **kwargs):
        super(KNetFx, self).__init__(filenames=filenames, **kwargs)

    def _define_model(self):
        input_ = Input(shape=(1, ))
        m_dense = kmden.(input_, self._hiddens, self._is_dropout)
        output = Dense(1)(m_dense)
        model = Model(input_, output, name='fx_model')
        self._model = model

    def _define_optimizer(self):
        self._optim = Adam(self._lr)

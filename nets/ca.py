"""Deep net for fx based on keras.
"""
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Nadam, Adam
from xlearn.knet.base import KNet
import xlearn.kmodel.dense as kmden
import xlearn.kmodel.image as kmcov

class KNetCa(KNet):

    def __init__(self, filenames=None, **kwargs):
        super(KNetCa, self).__init__(filenames=filenames, **kwargs)        
        self._n_conv_layer = self._settings['n_conv_layer']
        self._n_conv_channel = self._settings['n_conv_channel']
        self._ksize = self._settings['ksize']
    def _define_model(self):
        input_ = Input(shape=(10, 10, 1))
        m0_conv = kmcov.conv_seq(input_, [self._n_conv_channel//2]*self._n_conv_layer, [self._ksize]*self._n_conv_layer, [self._ksize]*self._n_conv_layer, id = 0)
        m1_conv = kmcov.conv_seq(m0_conv, [self._n_conv_channel]*self._n_conv_layer, [self._ksize]*self._n_conv_layer, [self._ksize]*self._n_conv_layer, id = 1)
        fla = Flatten()(m1_conv)
        m_dense = kmden.dense(fla, self._hiddens, self._is_dropout)
        output = Dense(3)(m_dense)
        model = Model(input_, output, name='fx_model')
        self._model = model

    def _define_optimizer(self):
        self._optim = Adam(self._lr)

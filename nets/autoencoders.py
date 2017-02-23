import tensorflow as tf
from keras.layers import Dense
from keras.objectives import mean_squared_error
from .base import Net
from ..utils.general import with_config
from ..model.layers import Input, Label


class AutoEncoder1D(Net):
    """ Auto encoder with a vector as input """

    @with_config
    def __init__(self, model_names=None, **kwargs):

        super(AutoEncoder1D, self).__init__(**kwargs)
        self._models_names = ['Autoencoder', 'encoder', 'decoder']
        self._is_train_step = [True, False, False]

    def _define_models(self):
        input_ = Input(shape=self._inputs_dims[0], name='input_vector')
        label = Label(shape=self._inputs_dims[0])
        encoded = Dense(self._hiddens[0], name='encoder')(input_)
        decoder = Dense(self._hiddens[1], name='decoder')
        decoded_data = decoder(encoded)
        input_code = Input((self._hiddens[0],), name='input_code')
        decoded_code = decoder(input_code)

        self._labels[0] = [label]
        self._inputs[0] = [input_]
        self._outputs[0] = [decoded_data]
        self._inputs[1] = [input_]
        self._outputs[1] = [encoded]
        self._inputs[2] = [input_code]
        self._outputs[2] = [decoded_code]

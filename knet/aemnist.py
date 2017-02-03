""" some autoencoders on mnist
[Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
"""
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from xlearn.knet.base import KNet


class AutoEncoder0(KNet):

    def __init__(self, **kwargs):
        super(AutoEncoder0, self).__init__(**kwargs)
        self._encoding_dim = self._settings.get("encoding_dim", 32)
        self._encoder = None
        self._decoder = None
        self._is_l1 = self._settings.get('is_l1', True)

    def _define_model(self):
        # this is our input placeholder
        input_img = Input(shape=(784,))
        # "encoded" is the encoded representation of the input
        if self._is_l1:
            encoded = Dense(self._encoding_dim, activation='relu',
                            activity_regularizer=regularizers.activity_l1(1e-3))(input_img)
        else:
            encoded = Dense(self._encoding_dim, activation='relu')(input_img)
        # "decoded" is the lossy reconstruction of the input
        decoder = Dense(784, activation='sigmoid')
        decoded = decoder(encoded)
        # this model maps an input to its reconstruction
        self._model = Model(input=input_img, output=decoded)
        self._encoder = Model(input=input_img, output=encoded)
        encoded_input = Input(shape=(self._encoding_dim,))
        decoded_layer = decoder(encoded_input)
        self._decoder = Model(input=encoded_input, output=decoded_layer)

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder

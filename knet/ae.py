import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, ELU, LeakyReLU, Convolution2D, UpSampling2D, MaxPooling2D, UpSampling2D, BatchNormalization
from keras.optimizers import Nadam, Adam, RMSprop

from .base import KNet, KGen

class AutoEncoder(KNet):

    def __init__(self, input_dim=(784,), **kwargs):
        super(AutoEncoderLSTM, self).__init__(input_dim=input_dim, **kwargs)
        self._input_dim = self._settings['input_dim']        
        self._encoder = None
        self._decoder = None
        self._nb_model = 3
    
    def get_model_id(self, name):        
        if name=='encoder':
            return 0
        if name=='decoder':
            return 1
        if name=='ae':
            return 2

    def _define_models(self):
        # this is our input placeholder
        input_ = Input(shape=(self._input_dim,))
        # "encoded" is the encoded representation of the input    
        encoded = Dense(self._encoding_dim, activation='relu')(input_)
        input_shifted = [0.0] + input_[:-1]
        # decoder = Dense(self._input_dim, activation='sigmoid')
        decoder = Dense(self._input_dim)
        decoded = decoder(encoded)
        # this model maps an input to its reconstruction
        self._models[self.get_model_id('ae')] = Model(input=input_img, output=decoded, name='AutoEncoder')
        self._models[self.get_model_id('encoder')] = Model(input=input_img, output=encoded, name='Encoder')
        encoded_input = Input(shape=(self._encoding_dim,))
        decoded_layer = decoder(encoded_input)
        self._models[self.get_model_id('decoder')] = Model(input=encoded_input, output=decoded_layer, name='Decoder')

    @property
    def autoencoder(self):
        return self._models[0]

    @property
    def encoder(self):
        return self._models[1]

    @property
    def decoder(self):
        return self._models[2]
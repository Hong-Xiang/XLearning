from xlearn.knet.base import KNet
from keras.models import Model
from keras.layers import Input, Dense

class AutoEncoderSimple(KNet):
    def __init__(self, **kwargs):
        super(AutoEncoderSimple, self).__init__(**kwargs)
        self._model_full = None
        self._model_encoder = None
        self._model_decoder = None


    def _define_model(self):
        x = Input(shape=[4])
        encoder = Dense(512, activation='relu')(x)
        z = Dense(2, name='code')(encoder)
        decoder = Dense(512, activation='relu')(z)
        y = Dense(4, name='recover')(decoder)
        model = Model(input=x, output=y, name='AutoEncoderSimple')
        self._model = model
        self._model_encoder = Model(input=x, output=z, name='encoder')

        # z_i = Input(shape=[2])
        # self._decoder = Model(input=z_i, output=, name='decoder')

    @property
    def encoder(self):
        return self._model_encoder

    @property
    def decoder(self):
        return self._model_decoder



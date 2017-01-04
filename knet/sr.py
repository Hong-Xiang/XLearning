from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.topology import Merge
from xlearn.knet.base import KNet
import xlearn.kmodel.image as kmi


class KNetSR(KNet):

    def __init__(self, filenames=None, **kwargs):
        super(KNetSR, self).__init__(filenames=filenames, **kwargs)
        self.shape_i = self._settings['shape_i']
        self.shape_o = self._settings['shape_o']
        self.batsh_size = self._settings['batch_size']
        self.n_upscale = self._settings['n_upscale']
        self.init = None
        self.n_residual = self._settings['n_residual']

    def _def_model(self):
        ip_lr = Input(
            shape=(self.shape_i[0], self.shape_i[1], 1), name='img_lr')
        ip_hr = Input(
            shape=(self.shape_o[0], self.shape_o[1], 1), name='img_hr')
        x = Convolution2D(64, 5, 5, border_mode='same',
                          name='sr_res_conv1', init=self.init)
        x = BatchNormalization(name='sr_res_bn_1')(x)
        x = LeakyReLU(name='sr_res_lr1')(x)
        for i in range(self.n_residual):
            x = kmi.residual_block(x, 3, 3, 3, i)
        for i in range(self.n_upscale):
            x = kmi.upscale_block(x, 3, i)
        x = Convolution2D(1, 5, 5, activation='tanh',
                          border_mode='same', name='sr_res_conv_final')(x)
        return x

    def _def_optimizer(self):
        self._optim = Adam(self._lr)

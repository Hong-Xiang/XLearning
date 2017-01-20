from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Nadam, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.topology import Merge
from xlearn.knet.base import KNet
import xlearn.kmodel.image as kmi


class KNetAE(KNet):

    def __init__(self, filenames=None, **kwargs):
        super(KNetAE, self).__init__(filenames=filenames, **kwargs)
        self.shape_i = self._settings['shape_i']
        self.shape_o = self._settings['shape_o']
        self.batsh_size = self._settings['batch_size']
        self.n_residual = self._settings['n_residual']
        self.n_hidden = self._settings['n_hidden']
        self.n_layer = self._settings['n_layer']

    def _define_model(self):
        print("DEFINT AutoEncode Net.")
        ip = Input(
            shape=(self.shape_i[0], self.shape_i[1], 1), name='img_input')
        x = Convolution2D(64, 5, 5, border_mode='same',
                          name='ae_conv_0')(ip)
        x = BatchNormalization(name='ae_bn_0')(x)
        x = LeakyReLU(name='ae_lr_0')(x)

        x = kmi.conv_seq(x, [64] * self.n_layer, [3] *
                         self.n_layer, [3] * self.n_layer, id=0)
        x = MaxPooling2D((3, 3), strides=(3, 3), name='max_pool_0')(x)

        x = kmi.conv_seq(x, [256] * self.n_layer, [3] *
                         self.n_layer, [3] * self.n_layer, id=1)
        x = MaxPooling2D((3, 3), strides=(3, 3), name='max_pool_1')(x)

        x = Convolution2D(512, 1, 1, border_mode='same',
                          name='ae_fc_0')(x)
        x = BatchNormalization(name='ae_bn_fc')(x)
        x = LeakyReLU(name='ae_lr_fc')(x)

        x = UpSampling2D(size=(3, 3), name='up_0')(x)
        x = BatchNormalization(name='ae_bn_up0')(x)
        x = LeakyReLU(name='ae_lr_up0')(x)
        x = Convolution2D(128, 1, 1, border_mode='same',
                          name='ae_cvbn_up_0')(x)
        x = BatchNormalization(name='ae_bn_up_0')(x)
        x = LeakyReLU(name='ae_cvlr_up_0')(x)

        x = UpSampling2D(size=(3, 3), name='up_1')(x)
        x = BatchNormalization(name='ae_bn_up1')(x)
        x = LeakyReLU(name='ae_lr_up1')(x)
        x = Convolution2D(64, 1, 1, border_mode='same',
                          name='ae_cv_up_1')(x)
        x = BatchNormalization(name='ae_cvbn_up_1')(x)
        x = LeakyReLU(name='ae_cvlr_up_1')(x)

        x = UpSampling2D(size=(3, 3), name='up_2')(x)
        x = BatchNormalization(name='ae_bn_up2')(x)
        x = LeakyReLU(name='ae_lr_up2')(x)
        x = Convolution2D(1, 5, 5, subsample=(3, 3), activation='tanh',
                          border_mode='same', name='sr_res_conv_final')(x)
        self._model = Model(input=ip, output=x)

    def _define_optimizer(self):
        self._optim = Adam(self._lr)

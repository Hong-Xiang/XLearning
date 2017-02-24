from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, merge, ELU, LeakyReLU, Convolution2D, UpSampling2D, BatchNormalization, Cropping2D
from keras.preprocessing.image import ImageDataGenerator
from keras.engine.topology import Merge
from xlearn.knet.base import KNet
import xlearn.kmodel.image as kmi
import xlearn.utils.xpipes as utp

from keras import backend as K
from keras.engine.topology import Layer
from xlearn.kmodel.image import convolution_block, convolution_blocks

from xlearn.reader.srinput import DataSetSuperResolution


def model_srcnn(input_, upratio, cropy, cropx):
    """ based on arxiv 1501.00092 """
    x = UpSampling2D(size=upratio)(input_)
    x = convolution_block(x, 64, 9, 9, id=0)
    x = convolution_block(x, 32, 1, 1, id=1)
    x = Convolution2D(1, 5, 5, border_mode='same')(x)
    return x


class KNetSR(KNet):

    def __init__(self, filenames=None, **kwargs):
        super(KNetSR, self).__init__(filenames=filenames, **kwargs)
        self._init = None

        self.shape_i = self._settings['shape_i']
        self.shape_o = self._settings['shape_o']
        self.downsample_ratio = self._settings['down_sample_ratio']
        self.n_residual = self._settings['n_resi_block']
        self.n_resi_c = self._settings['n_resi_channel']
        self.n_resi_l = self._settings['n_resi_layer']
        self.n_hidden = self._settings['n_hidden']
        self.n_layer = self._settings['n_layer']

    def _define_model(self):
        ip_lr = Input(
            shape=(self.shape_i[0], self.shape_i[1], 1), name='img_lr')
        x = model_srcnn(ip_lr, self.downsample_ratio[:2], 0, 0)
        self._model = Model(input=ip_lr, output=x)


if __name__ == "__main__":
    data_train = DataSetSuperResolution(
        filenames=['./super_resolution.json', './sino_train.json'])
    data_test = DataSetSuperResolution(
        filenames=['./super_resolution.json', './sino_test.json'])

    net = KNetSR(filenames=['./sino_train.json',
                            './super_resolution.json', './netsr.json'])
    net.define_net()
    net.model.fit_generator(data_train, samples_per_epoch=1024,
                            nb_epoch=100, callbacks=net.callbacks, validation_data=data_test, nb_val_samples=1024)
    x, y = data_test.next_batch()
    p = net.model.predict(x)
    np.save('x.npy', x)
    np.save('y.npy', y)
    np.save('p.npy', p)
    plot(net.model, to_file="model.png")

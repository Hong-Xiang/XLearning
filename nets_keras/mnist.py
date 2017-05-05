"""Deep classification network on Keras for MNIST
"""
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Activation, Convolution2D, MaxPooling2D, Dropout, Reshape
from keras.optimizers import Nadam, Adam, SGD
from xlearn.knet.base import KNet
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import xlearn.kmodel.dense as kmden
from keras.utils.visualize_util import plot


def model_squential_basic():
    """ one full connect layer model """
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(10, activation='softmax'))
    return model


def model_convolution():
    """ convolution based model """
    model = Sequential()
    model.add(Reshape((28, 28, 1), input_shape=(28, 28)))
    model.add(Convolution2D(32, 5, 5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 5, 5))
    model.add(MaxPooling2D((2, 2), (2, 2)))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model

 
class KNetMNIST(KNet):

    def __init__(self, filenames=None, **kwargs):
        super(KNetMNIST, self).__init__(filenames=filenames, **kwargs)

    def _define_model(self):
        if self._architecture == 'basic':
            self._model = model_squential_basic()
        else:
            self._model = model_convolution()

    def _define_optimizer(self):
        # self._optim = SGD(self._lr)
        self._optim = Adam(self._lr)
        # self._optim = Nadam(self._lr)
        # self._optim = Nadam()


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train = (x_train) / 128.0
    # x_test = (x_test) / 128.0
    # x_train -= - 1.0
    # x_test -= 1.0

    l_train = to_categorical(y_train, nb_classes=10)
    l_test = to_categorical(y_test, nb_classes=10)
    net = KNetMNIST(lr_init=1e-3, loss='categorical_crossentropy',
                    metrics=['accuracy'], architecture='default')
    net.define_net()
    for i in range(10):
        net.model.fit(x_train, l_train, batch_size=1024, nb_epoch=1,
                    validation_data=(x_test, l_test))
        print(net.lr)
    p = net.model.predict(x_test[:10])
    # model = squential_basic()
    # model.fit(x_train, l_train, batch_size=100, nb_epoch=10,
    #           validation_data=(x_test, l_test))
    # p = model.predict(x_train[:10])
    np.save('x.npy', x_test[:10, :, :])
    np.save('y.npy', y_test[:10])
    np.save('p.npy', p)
    plot(net.model, to_file="model.png")

from keras.datasets import mnist
from xlearn.knet.aemnist import AutoEncoder0
from xlearn.knet.aemnist import VariationalAutoEncoder
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
n_sample_train = len(x_train)
n_sample_test = len(x_test)
n_pixel = 784
intensity = 100
x_train *= intensity
x_test *= intensity

x_train_noise = x_train.reshape((np.prod(x_train.shape),))
x_train_noise = np.random.poisson(x_train_noise, size=x_train_noise.shape)

x_test_noise = x_test.reshape((np.prod(x_test.shape),))
x_test_noise = np.random.poisson(x_test_noise, size=x_test_noise.shape)


x_train = x_train.reshape((n_sample_train, n_pixel))
x_test = x_test.reshape((n_sample_test, n_pixel))
x_train_noise = x_train_noise.reshape((n_sample_train, n_pixel))
x_test_noise = x_test_noise.reshape((n_sample_test, n_pixel))

x_train_noise = x_train_noise.astype('float32')
x_test_noise = x_test_noise.astype('float32')

x_train /= intensity
x_test /= intensity
x_train_noise /= intensity
x_test_noise /= intensity


batch_size = 128

nb_batch_train = n_sample_train // batch_size
nb_batch_test = n_sample_test // batch_size
x_train = x_train[:nb_batch_train * batch_size, :]
x_test = x_test[:nb_batch_test * batch_size, :]
x_train_noise = x_train_noise[:nb_batch_train * batch_size, :]
x_test_noise = x_test_noise[:nb_batch_test * batch_size, :]

print(x_train.shape)
print(x_test.shape)
print(x_train_noise.shape)
print(x_test_noise.shape)


net = VariationalAutoEncoder(
    lr_init=1e-3, hiddens=[256, ], batch_size=batch_size, optim_name="RMSProp")
net.define_net()
net.model.fit(x_train_noise, x_train,
              nb_epoch=1500,
              batch_size=batch_size,
              shuffle=True,
              validation_data=(x_test, x_test))

decoded_imgs = net.model.predict(
    x_test_noise[:batch_size, :], batch_size=batch_size)

np.save('x_train.npy', x_train[:64, :])
np.save('x_test.npy', x_test[:64, :])
np.save('x_train_noise.npy', x_train_noise[:64, :])
np.save('x_test_noise.npy', x_test_noise[:64, :])
np.save('decoded_imgs.npy', decoded_imgs)
np.save('noise_images.npy', x_test_noise[:batch_size, :])

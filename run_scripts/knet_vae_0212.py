import numpy as np
import matplotlib.pyplot as plt

from xlearn.knet.aemnist import VariationalAutoEncoder
from xlearn.knet.aemnist import AutoEncoder
from xlearn.dataset.mnist import MNIST


def test_vae():
    BATCH_SZ = 128
    dataset_train = MNIST(is_noise=True, noise_scale=5.0, noise_type='poisson', is_unsp=True, is_flatten=True, is_norm=True, is_batch=True, batch_size=BATCH_SZ)
    dataset_test = MNIST(is_train=False, is_noise=True, noise_scale=5.0, noise_type='poisson', is_unsp=True, is_flatten=True, is_norm=True, is_batch=True, batch_size=BATCH_SZ)
    s = next(dataset_train)
    imgs_d = dataset_train.visualize(s, image_type='data')
    imgs_l = dataset_train.visualize(s, image_type='label')
    plt.figure(figsize=(8, 8))
    uti.subplot_images((imgs_d[:32], imgs_l[:32]), is_gray=True)
    s = next(dataset_train)
    imgs_d = dataset_train.visualize(s, image_type='data')
    imgs_l = dataset_train.visualize(s, image_type='label')
    plt.figure(figsize=(8, 8))
    uti.subplot_images((imgs_d[:32], imgs_l[:32]), is_gray=True)
    path_saves=['./save/ae','./save/en','./save/de']
    net = VariationalAutoEncoder(lrs=[1e-4], optims_names=['rmsporp'], hiddens=[256], batch_size=BATCH_SZ, path_saves=path_saves, path_loads=path_saves, sigma=1.0, encoding_dim=32)
    # net = AutoEncoder(lrs=[1e-4], optims_names=['rmsporp'])
    net.define_net()
    samples_per_epoch = 60000
    net.autoencoder.fit_generator(dataset_train, samples_per_epoch=samples_per_epoch, nb_epoch=100, validation_data=dataset_test, nb_val_samples=1024, callbacks=net.callbacks[0])
    test_sample = next(dataset_test)
    predict = net.autoencoder.predict(test_sample[0], batch_size=BATCH_SZ)
    formated_sample = (predict, test_sample[1], 1.0)
    img_d = dataset_test.visualize(test_sample, image_type='data')
    img_l = dataset_test.visualize(test_sample, image_type='label')
    img_p = dataset_test.visualize(formated_sample, image_type='data')
    plt.figure(figsize=(12, 8))
    uti.subplot_images((img_d[:32], img_l[:32], img_p[:32]), is_gray=True)
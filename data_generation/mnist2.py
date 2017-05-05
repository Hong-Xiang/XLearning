""" Modified MNIST2 dataset,
Put MNIST images into one of two possible position of image.
Image size: 56 x 28
With label: True
Dataset.format: TFRecords
Data.format:    unit8
"""
import numpy as np
from keras.datasets import mnist
import tensorflow as tf
from tqdm import tqdm
import h5py


def generate_dataset():
    (x1_train, _), (x1_test, _) = mnist.load_data()
    nb_train = x1_train.shape[0]
    nb_test = x1_test.shape[0]
    x_train = np.zeros(shape=[nb_train * 4, 28 * 2,
                              28 * 2, 1], dtype=np.float16)
    x_test = np.zeros(shape=[nb_train * 4, 28 * 2,
                             28 * 2, 1], dtype=np.float16)
    y_train = np.zeros(shape=[nb_train * 4, 28, 28, 1], dtype=np.float16)
    y_test = np.zeros(shape=[nb_train * 4, 28, 28, 1], dtype=np.float16)
    z_train = np.zeros(shape=[nb_train * 4], dtype=np.int)
    z_test = np.zeros(shape=[nb_train * 4], dtype=np.int)

    x1_train = x1_train / 255.0
    x1_test = x1_test / 255.0
    x1_train = x1_train.astype(np.float16)
    x1_train = x1_train.reshape((nb_train, 28, 28))
    x1_test = x1_test.astype(np.float16)
    x1_test = x1_test.reshape((nb_test, 28, 28))
    for i in tqdm(range(nb_train)):
        x_train[4 * i, 0:28, 0:28, 0] = x1_train[i, :, :]
        y_train[4 * i, ..., 0] = x1_train[i, :, :]
        z_train[4 * i] = 0
        x_train[4 * i + 1, 28:, 0:28, 0] = x1_train[i, :, :]
        y_train[4 * i + 1, ..., 0] = x1_train[i, :, :]
        z_train[4 * i + 1] = 1
        x_train[4 * i + 2, 0:28, 28:, 0] = x1_train[i, :, :]
        y_train[4 * i + 2, ..., 0] = x1_train[i, :, :]
        z_train[4 * i + 2] = 2
        x_train[4 * i + 3, 28:, 28:, 0] = x1_train[i, :, :]
        y_train[4 * i + 3, ..., 0] = x1_train[i, :, :]
        z_train[4 * i + 3] = 3
    for i in tqdm(range(nb_test)):
        x_test[4 * i, 0:28, 0:28, 0] = x1_test[i, :, :]
        z_test[4 * i] = 0
        y_test[4 * i, ..., 0] = x1_test[i, ...]
        x_test[4 * i + 1, 28:, 0:28, 0] = x1_test[i, :, :]
        y_test[4 * i + 1, ..., 0] = x1_test[i, ...]
        z_test[4 * i + 1] = 1
        x_test[4 * i + 2, 0:28, 28:, 0] = x1_test[i, :, :]
        y_test[4 * i + 2, ..., 0] = x1_test[i, ...]
        z_test[4 * i + 2] = 2
        x_test[4 * i + 3, 28:, 28:, 0] = x1_test[i, :, :]
        y_test[4 * i + 3, ..., 0] = x1_test[i, ...]
        z_test[4 * i + 3] = 3
    x_train -= 0.5
    x_test -= 0.5
    y_train -= 0.5
    y_test -= 0.5
    return (x_train, y_train, z_train), (x_test, y_test, z_test)


def save_dataset(data_train, data_test):
    nb_train = data_train[0].shape[0]
    nb_test = data_test[0].shape[0]
    with h5py.File('mnist2.h5', 'w') as fout:
        trainset = fout.create_group('train')
        testset = fout.create_group('test')
        train_data = trainset.create_dataset(
            'image', data=data_train[0], dtype=np.float16)
        train_data = trainset.create_dataset(
            'label', data=data_train[1], dtype=np.float16)
        train_label = trainset.create_dataset(
            'latent', data=data_train[2], dtype=np.int)
        test_data = testset.create_dataset(
            'image', data=data_test[0], dtype=np.float16)
        test_label = testset.create_dataset(
            'label', data=data_test[1], dtype=np.float16)
        test_label = testset.create_dataset(
            'latent', data=data_test[2], dtype=np.int)
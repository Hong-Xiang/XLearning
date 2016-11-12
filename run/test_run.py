#!/usr/bin/python
from __future__ import absolute_import, division, print_function
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import xlearn.nets.layers
import argparse
import os

from xlearn.reader.srinput import DataSet
from xlearn.model.supernet import *
from xlearn.nets.model import NetManager
from xlearn.reader.oneoverx import DataSetOneOverX

FLAGS = tf.app.flags.FLAGS


def check_dataset(dataset):
    data, label = dataset.next_batch()
    n_show = 4
    for i in range(n_show):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(data[i, :, :, 0])
        plt.gray()
        plt.show()
        plt.subplot(1, 2, 2)
        plt.imshow(label[i, :, :, 0])
        plt.gray()
        plt.show()

def test_one_over_x(argv):
    data_set = DataSetOneOverX(batch_size=FLAGS.batch_size)
    net = xlearn.model.oneoverx.NetOneOverX(batch_size=FLAGS.batch_size)
    manager = NetManager(net)
    n_step = FLAGS.steps
    for i in range(n_step):
        x, y = data_set.next_batch()
        [loss_train, _, lr] = manager.run([net.loss, net.train, net.learn_rate],
                                          feed_dict={net.inputs: x, net.label: y})
        if i % 10 == 0:
            print('step={0:5d},\tlr={2:.3E},\t loss={1:4E}.'.format(
                i, loss_train, lr))
        if i % 20 == 0:
            manager.write_summary(
                feed_dict={net.inputs: x, net.label: y})
        if i % 50 == 0:
            x, y = data_set.next_batch()
            [loss_test] = manager.run([net.loss],
                                      feed_dict={net.inputs: x,
                                                 net.label: y})
            print('step={0:5d},\t test loss={1:4E}.'.format(i, loss_test))
    
    x = np.linspace(0.1, 1.1, num=FLAGS.batch_size)
    x = np.reshape(x, [FLAGS.batch_size, 1])
    y_ = manager.run([net.infer], feed_dict={net.inputs: x})
    np.save('oneoverx.npy', y_)


def testSR(argv):
    patch_shape = [FLAGS.height * FLAGS.down_ratio,
                   FLAGS.width * FLAGS.down_ratio]
    strides = [1, 1]
    train_set = DataSet(path=FLAGS.train_path,
                        prefix=FLAGS.prefix,
                        patch_shape=patch_shape, strides=strides,
                        batch_size=FLAGS.batch_size,
                        n_patch_per_file=FLAGS.patch_per_file,
                        down_sample_ratio=FLAGS.down_ratio,
                        dataset_type='train',
                        down_sample_method='mean')

    test_set = DataSet(path=FLAGS.test_path,
                       prefix=FLAGS.prefix,
                       patch_shape=patch_shape, strides=strides,
                       batch_size=FLAGS.batch_size,
                       n_patch_per_file=FLAGS.patch_per_file,
                       down_sample_ratio=FLAGS.down_ratio,
                       dataset_type='test',
                       down_sample_method='mean')

    # check_dataset(train_set)
    # check_dataset(test_set)

    if FLAGS.task == "SuperNetCrop":
        net = xlearn.model.supernet.SuperNetCrop()
    if FLAGS.task == "SuperNet0":
        net = xlearn.model.supernet.SuperNet0()
    if FLAGS.task == "SuperNet1":
        net = xlearn.model.supernet.SuperNet1()
    if FLAGS.task == "SuperNet2":
        net = xlearn.model.supernet.SuperNet1()
    manager = NetManager(net)

    n_step = FLAGS.steps
    for i in range(n_step):
        data, label = train_set.next_batch()
        [loss_train, _, lr] = manager.run([net.loss, net.train, net.learn_rate],
                                          feed_dict={net.inputs: data, net.label: label})
        if i % 10 == 0:
            print('step={0:5d},\tlr={2:.3E},\t loss={1:0.3f}.'.format(
                i, loss_train, lr))
        if i % 20 == 0:
            manager.write_summary(
                feed_dict={net.inputs: data, net.label: label})
        if i % 50 == 0:
            data_test, label_test = test_set.next_batch()
            [loss_test] = manager.run([net.loss],
                                      feed_dict={net.inputs: data_test,
                                                 net.label: label_test})
            print('step={0:5d},\t test loss={1:0.3f}.'.format(i, loss_test))
    # manager.save()

    saver = tf.train.Saver(tf.all_variables())
    path = saver.save(manager.sess, FLAGS.save_path, FLAGS.steps)
    print(path)


def main(argv):
    testSR(argv)
    #test_one_over_x(argv)

if __name__ == '__main__':
    xlearn.nets.model.define_flags()
    xlearn.nets.model.before_net_definition()
    tf.app.run()

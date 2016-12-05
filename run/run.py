#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import xlearn.nets.layers
import xlearn.utils.xpipes as utp
import argparse
import os

from xlearn.model.supernet import *
from xlearn.nets.model import NetManager
from xlearn.reader.oneoverx import DataSetOneOverX
from xlearn.reader.srinput import DataSetSR
from xlearn.reader.srinput import DataSetSRInfer

import xlearn.utils.image as uti
import xlearn.utils.tensor as utt

FLAGS = tf.app.flags.FLAGS


def construct_net():
    if FLAGS.net == "SuperNetInterp":
        net = xlearn.model.supernet.SuperNetInterp()
    if FLAGS.net == "SuperNetCrop":
        net = xlearn.model.supernet.SuperNetCrop()
    if FLAGS.net == "SuperNet0":
        net = xlearn.model.supernet.SuperNet0()
    if FLAGS.net == "SuperNet1":
        net = xlearn.model.supernet.SuperNet1()
    if FLAGS.net == "SuperNet2":
        net = xlearn.model.supernet.SuperNet2()
    return net


def construct_dataset(conf_file):
    if FLAGS.task == "train_SR_sino" or FLAGS.task == "train_SR_nature":
        data_set = DataSetSR(conf=conf_file)
    if FLAGS.task == "infer_SR_Sino":
        data_set = DataSetSRInfer(conf_file)
    return data_set


def check_dataset(dataset, n_show=4):
    data, label = dataset.next_batch()
    for i in range(n_show):
        plt.subplot(n_show, 2, i * 2 + 1)
        plt.imshow(data[i, :, :, 0])
        plt.gray()
        plt.subplot(n_show, 2, i * 2 + 2)
        plt.imshow(label[i, :, :, 0])
        plt.gray()
    plt.show()


def train_one_over_x(argv):
    """learn: y = 1/x
    """
    data_set = DataSetOneOverX(batch_size=FLAGS.batch_size)
    net = xlearn.model.oneoverx.NetOneOverX(batch_size=FLAGS.batch_size)
    manager = NetManager(net)
    n_step = FLAGS.steps
    for i in range(1, n_step + 1):
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

    # save test result with x and y
    x = np.linspace(0.1, 1.1, num=FLAGS.batch_size)
    x = np.reshape(x, [FLAGS.batch_size, 1])
    y_ = manager.run([net.infer], feed_dict={net.inputs: x})
    np.save('oneoverx.npy', [x, y_])


def infer_SR_Sino(argv):
    dataset_infer = DataSetSRInfer(FLAGS.infer_conf)
    infer_path = dataset_infer.path_infer
    net = construct_net()
    manager = NetManager(net)
    manager.restore()
    pipe_file = utp.FileNameLooper(infer_path, prefix='sino')
    for input_file in pipe_file.out:
        dataset_infer.load_new_file(input_file)
        input_file = os.path.basename(input_file)
        for i in xrange(dataset_infer.n_batch):
            low_sr = dataset_infer.next_batch()
            result = manager.run(net.infer, feed_dict={net.inputs: low_sr})
            dataset_infer.add_result(result)
        dataset_infer.form_image()
        image_orginal = dataset_infer.image
        image_superre = dataset_infer.recon
        output_name = 'sr' + input_file
        input_name = 'or' + input_file
        fullnameo = os.path.join(dataset_infer.path_output, output_name)
        fullnamei = os.path.join(dataset_infer.path_output, input_name)
        np.save(fullnameo, image_superre)
        np.save(fullnamei, image_orginal)
    
    # image_orginal = uti.image_formater(image_orginal)

    # image_superre = uti.image_formater(image_superre)


def train_SR_sino(argv):
    """train super resolution net on sinogram 2d data.
    """
    train_set = construct_dataset(FLAGS.train_conf)
    test_set = construct_dataset(FLAGS.test_conf)
    # check_dataset(train_set)
    # check_dataset(test_set)
    # return None
    net = construct_net()
    manager = NetManager(net)
    test_loss = []
    n_step = FLAGS.steps
    saver = tf.train.Saver(tf.all_variables())
    for i in range(1, n_step + 1):
        data, label = train_set.next_batch()
        [loss_train, _, lr] = manager.run([net.loss, net.train, net.learn_rate],
                                          feed_dict={net.inputs: data, net.label: label})
        if i % 10 == 0:
            print('step={0:5d},\tlr={2:.3E},\t loss={1:0.3f}.'.format(
                i, loss_train, lr))
        if i % 20 == 0:
            manager.write_summary(
                feed_dict={net.inputs: data, net.label: label})
            data_test, label_test = test_set.next_batch()
            manager.write_summary(
                feed_dict={net.inputs: data_test, net.label: label_test}, is_test=True)
        if i % 50 == 0:
            data_test, label_test = test_set.next_batch()
            [loss_test] = manager.run([net.loss],
                                      feed_dict={net.inputs: data_test,
                                                 net.label: label_test})
            print('step={0:5d},\t test loss={1:0.3f}.'.format(i, loss_test))
        if i % 1000 == 0:
            path = saver.save(manager.sess, FLAGS.save_path, i)        
    # manager.save()
    np.save('test_loss.npy', np.array(test_loss))    
    path = saver.save(manager.sess, FLAGS.save_path, FLAGS.steps)
    print("net variables saved to: " + path + '.')


def train_SR_nature(argv):
    train_set = construct_dataset(FLAGS.train_conf)
    test_set = construct_dataset(FLAGS.test_conf)
    # check_dataset(train_set)
    # check_dataset(test_set)
    # return None

    net = construct_net()
    manager = NetManager(net)

    n_step = FLAGS.steps
    for i in range(1, n_step + 1):
        data, label = train_set.next_batch()
        [loss_train, _, lr] = manager.run([net.loss, net.train, net.learn_rate],
                                          feed_dict={net.inputs: data, net.label: label})
        if i % 10 == 0:
            print('step={0:5d},\tlr={2:.3E},\t loss={1:0.3f}.'.format(
                i, loss_train, lr))
        if i % 20 == 0:
            manager.write_summary(
                feed_dict={net.inputs: data, net.label: label})
            data_test, label_test = test_set.next_batch()
            manager.write_summary(
                feed_dict={net.inputs: data_test, net.label: label_test}, is_test=True)
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
    if FLAGS.task == "train_one_over_x":
        train_one_over_x(argv)
    if FLAGS.task == "train_SR_nature":
        train_SR_nature(argv)
    if FLAGS.task == "train_SR_sino":
        train_SR_sino(argv)
    if FLAGS.task == "infer_SR_sino":
        infer_SR_Sino(argv)


if __name__ == '__main__':
    xlearn.nets.model.define_flags(sys.argv[1])
    xlearn.nets.model.before_net_definition()
    tf.app.run()

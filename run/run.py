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

FLAGS = tf.app.flags.FLAGS


def construct_net():
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
    # patch_shape = [FLAGS.height,
    #                FLAGS.width * FLAGS.down_ratio]
    # strides = [5, 5]

    # fullname = os.path.join(FLAGS.infer_path, FLAGS.infer_file)
    # input_ = np.load(fullname)
    # pipe_input = utp.Inputer()
    # pipe_input.insert(input_)
    # pipe_patch = utp.PatchGenerator(pipe_input, patch_shape, strides)
    # pipe_patch_tensor = utp.TensorFormater(pipe_patch)
    # pipe_down_sample = utp.DownSamplerSingle(
    #     pipe_patch_tensor, axis=2, ratio=FLAGS.down_ratio, method='fixed')
    # pipe_sliced = utp.TensorSlicer(pipe_down_sample)
    # input_list = pipe_sliced.out.next()

    # len_list = len(input_list)
    # n_batch = int(np.ceil(float(len_list) / FLAGS.batch_size))

    # net = construct_net()
    # manager = NetManager(net)

    # patch_shape_down = input_list[0].shape
    # patch_result = []
    # valid_offset = [FLAGS.hidden_layer, FLAGS.hidden_layer]

    # mean_std = 0
    # for i in xrange(len_list):
    #     mean_std += np.std(input_list[i])
    # mean_std /= len_list
    # mean_mean = 0
    # for i in xrange(len_list):
    #     input_list[i] /= mean_std
    #     mean_mean += np.mean(input_list[i])
    # mean_mean /= len_list
    # for i in xrange(len_list):
    #     input_list[i] -= mean_mean
    # cid = 0
    # for i in xrange(n_batch):
    #     tensor_input = np.zeros(
    #         [FLAGS.batch_size, patch_shape_down[0], patch_shape_down[1], 1])
    #     for j in xrange(FLAGS.batch_size):
    #         if cid < len_list:
    #             temp = input_list[cid]
    #         else:
    #             temp = np.zeros(patch_shape_down)
    #         cid += 1
    #         tensor_input[j, :, :, 0] = temp
    #     tensor_output = manager.run(
    #         net.infer, feed_dict={net.inputs: tensor_input})
    #     for j in xrange(FLAGS.batch_size):
    #         result = tensor_output[j, :, :, 0]
    #         result += mean_mean
    #         result *= mean_std
    #         result_pad = np.zeros([1] + patch_shape + [1])
    #         result_pad[0, valid_offset[0]:-valid_offset[0],
    #                    valid_offset[1]:-valid_offset[1], 0] = result
    #         patch_result.append(result_pad)
    # patch_result = patch_result[:len_list]
    # output = xlearn.utils.tensor.patches_recon_tensor(
    #     patch_result, input_.shape, patch_shape, strides, [23, 89], valid_offset)
    # np.save(FLAGS.infer_file, output)

    # print(tensor.shape)
    pass


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
    np.save('test_loss.npy', np.array(test_loss))
    saver = tf.train.Saver(tf.all_variables())
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
    # inferSino(argv)


if __name__ == '__main__':
    xlearn.nets.model.define_flags(sys.argv[1])
    xlearn.nets.model.before_net_definition()
    tf.app.run()

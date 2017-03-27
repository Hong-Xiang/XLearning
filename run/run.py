#!/usr/bin/python
# -*- coding: utf-8 -*-
"""entry script for all runs.
"""
# TODO: add a new script dedicate for command line interaction.
import matplotlib.pyplot as plt
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
from xlearn.reader.fx import DataSetFx
from xlearn.reader.srinput import DataSetSuperResolution
import xlearn.reader.srinput

import xlearn.utils.image as uti
import xlearn.utils.tensor as utt
import xlearn.utils.general as utg

FLAGS = tf.app.flags.FLAGS


def construct_net(filenames=None, **kwargs):
    if FLAGS.net_name == "SuperNetInterp":
        net = xlearn.model.supernet.SuperNetInterp(
            filenames=filenames, **kwargs)
    if FLAGS.net_name == "SuperNetCrop":
        net = xlearn.model.supernet.SuperNetCrop(filenames=filenames, **kwargs)
    if FLAGS.net_name == "SuperNet0":
        net = xlearn.model.supernet.SuperNet0(filenames=filenames, **kwargs)
    if FLAGS.net_name == "SuperNet1":
        net = xlearn.model.supernet.SuperNet1(filenames=filenames, **kwargs)
    if FLAGS.net_name == "SuperNet2":
        net = xlearn.model.supernet.SuperNet2(filenames=filenames, **kwargs)
    if FLAGS.net_name == "NetFx":
        net = xlearn.model.fx.NetFx(filenames=filenames, **kwargs)
    if FLAGS.net_name == "NetFxNoise":
        net = xlearn.model.fx.NetFxNoise(filenames=filenames, **kwargs)
    return net


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


def train_fx(argv):
    """learn: y = 1/x or general fx
    """
    data_set = DataSetFx(filenames=argv[2:], func=lambda x: 1 / x)
    net = xlearn.model.fx.NetFx(filenames=argv[2:])
    manager = NetManager(net)

    n_step = FLAGS.run_steps
    for i in range(1, n_step + 1):
        x, y = data_set.next_batch()
        [loss_train, _, lr] = manager.run([net.loss, net.train, net.learn_rate],
                                          feed_dict={net.inputs: x, net.label: y, net.keep_prob: 0.5})
        if i % 10 == 0:
            print('step={0:5d},\tlr={2:.3E},\t loss={1:4E}.'.format(
                i, loss_train, lr))
        if i % 20 == 0:
            manager.write_summary(
                feed_dict={net.inputs: x, net.label: y, net.keep_prob: 0.5})
        if i % 50 == 0:
            x, y = data_set.next_batch()
            [loss_test] = manager.run([net.loss],
                                      feed_dict={net.inputs: x,
                                                 net.label: y,
                                                 net.keep_prob: 0.5})
            print('step={0:5d},\t test loss={1:4E}.'.format(i, loss_test))

    # save test result with x and y
    x = np.linspace(data_set.xmin, data_set.xmax, num=data_set.batch_size)
    x = np.reshape(x, [data_set.batch_size, 1])
    y_ = manager.run(net.infer, feed_dict={net.inputs: x, net.keep_prob: 1.0})
    np.save('oneoverx.npy', [x, y_])


def infer_super_resolution(argv):
    """inference with super resolution net
    """
    dataset_infer = DataSetSuperResolution(argv[2:])
    dataset_infer.free_lock()
    datarecon = xlearn.reader.srinput.ImageReconstructer(argv[2:])
    net = construct_net(filenames=argv[1:])
    manager = NetManager(net)
    manager.restore()
    for i in range(dataset_infer.n_files):
        print("Processing {0} of {1} files...".format(
            i, dataset_infer.n_files))
        dataset_infer.free_lock()
        while True:
            try:
                low_sr, _ = dataset_infer.next_batch()
                result = manager.run(net.infer, feed_dict={net.inputs: low_sr})
                if dataset_infer.norm_method == "patch":
                    means, stds = dataset_infer.moments()
                    datarecon.add_result(result, means=means, stds=stds)
                else:
                    datarecon.add_result(result)
            except xlearn.reader.base.NoMoreEpoch:
                break
            except xlearn.reader.base.NoMoreSample:
                break

        input_file = os.path.basename(dataset_infer.last_file)
        image = datarecon.reconstruction()
        output_file = os.path.join(datarecon.path_output, input_file)
        np.save(output_file, image)


def train_super_resolution(argv):
    """train super resolution net
    """
    train_files = [argv[2], argv[3]]
    test_files = [argv[2], argv[4]]
    train_set = DataSetSuperResolution(filenames=train_files)
    test_set = DataSetSuperResolution(filenames=test_files)
    net_files = argv[1:]
    net = construct_net(filenames=net_files)
    logging.debug('net constructed.')
    manager = NetManager(net)
    logging.debug('manager constructed.')
    n_step = FLAGS.run_steps
    saver = tf.train.Saver(tf.global_variables())
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
            path = saver.save(manager.sess, FLAGS.path_save, i)
    manager.save()
    # np.save('test_loss.npy', np.array(test_loss))
    # path = saver.save(manager.sess, FLAGS.path_save, FLAGS.run_steps)
    print("net variables saved to: " + path + '.')


def main(argv):
    if FLAGS.run_task == "train_fx":
        train_fx(argv)
    elif FLAGS.run_task == "train_super_resolution":
        train_super_resolution(argv)
    elif FLAGS.run_task == "infer_super_resolution":
        infer_super_resolution(argv)
    else:
        raise ValueError("Unkown task.")


if __name__ == '__main__':
    xlearn.nets.model.define_flags(sys.argv[1])
    xlearn.nets.model.before_net_definition()
    tf.app.run()

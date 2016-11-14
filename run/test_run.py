#!/usr/bin/python
from __future__ import absolute_import, division, print_function
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import xlearn.nets.layers
import xlearn.utils.xpipes as xp
import argparse
import os

from xlearn.reader.srinput import DataSet
from xlearn.model.supernet import *
from xlearn.nets.model import NetManager
from xlearn.reader.oneoverx import DataSetOneOverX
from xlearn.reader.sino import DataSetSinogram

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

def inferSino(argv):
    patch_shape = [FLAGS.height,
                   FLAGS.width * FLAGS.down_ratio]
    strides = [5, 5]

    fullname = os.path.join(FLAGS.infer_path, FLAGS.infer_file)
    input_ = np.load(fullname)    
    pipe_input = xp.Inputer()
    pipe_input.insert(input_)
    pipe_patch = xp.PatchGenerator(pipe_input, patch_shape, strides)
    pipe_patch_tensor = xp.TensorFormater(pipe_patch)
    pipe_down_sample = xp.DownSamplerSingle(pipe_patch_tensor, axis=2, ratio=FLAGS.down_ratio, method='fixed')
    pipe_sliced = xp.TensorSlicer(pipe_down_sample)
    input_list = pipe_sliced.out.next()
    
    len_list = len(input_list)
    n_batch = int(np.ceil( float(len_list)/FLAGS.batch_size ))


    if FLAGS.task == "SuperNetCrop":
        net = xlearn.model.supernet.SuperNetCrop()
    if FLAGS.task == "SuperNet0":
        net = xlearn.model.supernet.SuperNet0()
    if FLAGS.task == "SuperNet1":
        net = xlearn.model.supernet.SuperNet1()
    if FLAGS.task == "SuperNet2":
        net = xlearn.model.supernet.SuperNet2()
    manager = NetManager(net)
    patch_shape_down = input_list[0].shape
    patch_result = []
    valid_offset = [FLAGS.hidden_layer, FLAGS.hidden_layer]
    
    mean_std = 0
    for i in xrange(len_list):
        mean_std += np.std(input_list[i])
    mean_std /= len_list
    mean_mean = 0
    for i in xrange(len_list):
        input_list[i] /= mean_std
        mean_mean += np.mean(input_list[i])
    mean_mean /= len_list
    for i in xrange(len_list):
        input_list[i] -= mean_mean
    cid = 0
    for i in xrange(n_batch):
        tensor_input = np.zeros([FLAGS.batch_size, patch_shape_down[0], patch_shape_down[1], 1])
        for j in xrange(FLAGS.batch_size):
            if cid < len_list:
                temp = input_list[cid]
            else:
                temp = np.zeros(patch_shape_down)
            cid += 1
            tensor_input[j, :, :, 0] = temp
        tensor_output = manager.run(net.infer, feed_dict={net.inputs: tensor_input})        
        for j in xrange(FLAGS.batch_size):
            result = tensor_output[j, :, :, 0]
            result += mean_mean
            result *= mean_std
            result_pad = np.zeros([1]+patch_shape+[1])
            result_pad[0, valid_offset[0]:-valid_offset[0], valid_offset[1]:-valid_offset[1], 0] = result            
            patch_result.append(result_pad)
    patch_result = patch_result[:len_list]
    output = xlearn.utils.tensor.patches_recon_tensor(patch_result, input_.shape, patch_shape, strides, [23, 89], valid_offset)
    np.save(FLAGS.infer_file,output)

    # print(tensor.shape)


def testSino(argv):
    patch_shape = [FLAGS.height,
                   FLAGS.width * FLAGS.down_ratio]
    strides = [1, 1]
    train_set = DataSetSinogram(path=FLAGS.train_path,
                                prefix=FLAGS.prefix,
                                patch_shape=patch_shape, strides=strides,
                                batch_size=FLAGS.batch_size,
                                n_patch_per_file=FLAGS.patch_per_file,
                                down_sample_ratio=FLAGS.down_ratio,
                                dataset_type='train',
                                down_sample_method='fixed')

    test_set = DataSetSinogram(path=FLAGS.test_path,
                               prefix=FLAGS.prefix,
                               patch_shape=patch_shape, strides=strides,
                               batch_size=FLAGS.batch_size,
                               n_patch_per_file=FLAGS.patch_per_file,
                               down_sample_ratio=FLAGS.down_ratio,
                               dataset_type='train',
                               down_sample_method='fixed')
    if FLAGS.task == "SuperNetCrop":
        net = xlearn.model.supernet.SuperNetCrop()
    if FLAGS.task == "SuperNet0":
        net = xlearn.model.supernet.SuperNet0()
    if FLAGS.task == "SuperNet1":
        net = xlearn.model.supernet.SuperNet1()
    if FLAGS.task == "SuperNet2":
        net = xlearn.model.supernet.SuperNet2()
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
        if i % 50 == 0:
            data_test, label_test = test_set.next_batch()
            [loss_test] = manager.run([net.loss],
                                      feed_dict={net.inputs: data_test,
                                                 net.label: label_test})
            print('step={0:5d},\t test loss={1:0.3f}.'.format(i, loss_test))
            test_loss.append(loss_test)
    # manager.save()
    np.save('test_loss.npy',np.array(test_loss))
    saver = tf.train.Saver(tf.all_variables())
    path = saver.save(manager.sess, FLAGS.save_path, FLAGS.steps)
    print("net variables saved to: " + path + '.')

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
        net = xlearn.model.supernet.SuperNet2()
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
    # testSR(argv)
    #test_one_over_x(argv)
    testSino(argv)
    # inferSino(argv)

if __name__ == '__main__':
    xlearn.nets.model.define_flags()
    xlearn.nets.model.before_net_definition()
    tf.app.run()

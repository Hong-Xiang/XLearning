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
from xlearn.model.supernet import SuperNet0
from xlearn.model.supernet import SuperNet1
from xlearn.nets.model import NetManager

FLAGS = tf.app.flags.FLAGS



def check_dataset(dataset):
    data, label= dataset.next_batch()
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

def testSR(argv):
    patch_shape = [FLAGS.height*FLAGS.down_ratio, FLAGS.width*FLAGS.down_ratio]
    strides = [5, 5]
    train_set = DataSet(path=FLAGS.train_path,
                        prefix=FLAGS.prefix,
                        patch_shape=patch_shape, strides=strides,
                        batch_size=FLAGS.train_batch_size,
                        n_patch_per_file=FLAGS.patch_per_file,
                        down_sample_ratio=FLAGS.down_ratio,
                        dataset_type='train',
                        down_sample_method='mean')

    test_set = DataSet(path=FLAGS.test_path,
                       prefix=FLAGS.prefix,
                       patch_shape=patch_shape, strides=strides,
                       batch_size=FLAGS.train_batch_size,
                       n_patch_per_file=FLAGS.patch_per_file,
                       down_sample_ratio=FLAGS.down_ratio,
                       dataset_type='test',
                       down_sample_method='mean')

    # check_dataset(train_set)
    # check_dataset(test_set)
    
    net = SuperNet1()
    manager = NetManager(net)    

    n_step = 3001
    for i in range(n_step):
        data, label = train_set.next_batch()                        
        [loss_train, _, lr] = manager.run([net.loss, net.train, net.learn_rate], feed_dict={net.inputs: data, net.label: label})
        if i%10 == 0:
            print('step={0:5d},\tlr={2:.3E},\t loss={1:.3E}.'.format(i, loss_train, lr))
        if i%20 == 0:
            manager.write_summary(feed_dict={net.inputs: data, net.label: label})
        if i%50 == 0:
            data_test, label_test = test_set.next_batch()            
            [loss_test] = manager.run([net.loss], feed_dict={net.inputs: data_test, net.label: label_test})
            print('step={0:5d},\t test loss={1:.3E}.'.format(i, loss_test))
    manager.save()           


def main(argv):
    testSR(argv)


if __name__ == '__main__':
    xlearn.nets.model.define_flags(sys.argv)
    xlearn.nets.model.before_net_definition()
    tf.app.run()

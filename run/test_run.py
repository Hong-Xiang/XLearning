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

def define_flags(argv):
    flag = tf.app.flags
    flag.DEFINE_float("weight_decay", 0.01, "Weight decay coefficient.")
    flag.DEFINE_float("eps", 1e-5, "Weight decay coefficient.")    
    flag.DEFINE_integer("train_batch_size", 64, "Batch size.")
    flag.DEFINE_integer("test_batch_size", 10, "Batch size.")    
    flag.DEFINE_integer("hidden_units", 1024, "# of hidden units.")
    flag.DEFINE_float("learning_rate_init", None, "Initial learning rate.")
    flag.DEFINE_string("save_dir",'.',"saving path.")
    flag.DEFINE_string("summary_dir",'.',"summary path.")
    flag.DEFINE_integer("decay_steps",100000,"decay steps.")
    flag.DEFINE_float("learning_rate_decay_factor",0.1,"learing rate decay factor.")
    flag.DEFINE_integer("height", 33, "patch_height")
    flag.DEFINE_integer("width", 33, "patch_width")
    flag.DEFINE_float("down_ratio",3,"down_sample_ratio")
    flag.DEFINE_integer("patch_per_file", 8, "patches per file.")    
    flag.DEFINE_string("train_path", None, "train data path.")
    flag.DEFINE_string("test_path", None, "train data path.")
    flag.DEFINE_string("prefix",None,"prefix of data files.")
    flag.DEFINE_float("leak_ratio_constant", None, "lrelu constant.")
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

def test(argv):
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

    n_step = 1001
    for i in range(n_step):
        data, label = train_set.next_batch()                        
        [loss_train, _, lr] = manager.run([net.loss, net.train, net.learn_rate], feed_dict={net.inputs:data, net.label:label})
        if i%1 == 0:
            print('step={0:5d},\tlr={2:.3E},\t loss={1:.3E}.'.format(i, loss_train, lr))
        if i%10 == 0:
            manager.write_summary(feed_dict={net.inputs:data, net.label:label})                    
        if i%50 == 0:
            data_test, label_test = test_set.next_batch()            
            [loss_test] = manager.run([net.loss], feed_dict={net.inputs:data_test, net.label:label_test})
            print('step={0:5d},\t test loss={1:.3E}.'.format(i, loss_test))
    manager.save()           
    
def main(argv):    
    test(argv)

if __name__=='__main__':
    define_flags(sys.argv)
    xlearn.nets.model.before_net_definition()
    tf.app.run()
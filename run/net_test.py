#!/usr/bin/python
from __future__ import absolute_import, division, print_function
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import xlearn.nets.layers
from xlearn.reader.mnist import DataSet
from xlearn.nets.model import MNIST
from tensorflow.examples.tutorials.mnist import input_data
FLAGS = tf.app.flags.FLAGS

def define_flags(argv):    
    flag = tf.app.flags
    flag.DEFINE_float("weight_decay", 0.0, "Weight decay coefficient.")
    flag.DEFINE_float("eps", 1e-5, "Weight decay coefficient.")    
    flag.DEFINE_integer("train_batch_size", 100, "Batch size.")
    flag.DEFINE_integer("test_batch_size", 10000, "Batch size.")
    flag.DEFINE_integer("hidden_units", 50, "Batch size.")

def check_dataset(dataset):
    data, label= dataset.next_batch()        
    n_show = 4
    for i in range(n_show):
        plt.figure()
        plt.imshow(data[i, :, :])
        plt.gray()
        plt.show()
        digit = list(label[i,:]==1).index(True)
        print(digit)

def test(argv):
    train_set = DataSet(data_file='/home/hongxwing/Workspace/Datas/MNIST/train_data.npy',
                        label_file='/home/hongxwing/Workspace/Datas/MNIST/train_label.npy',
                        is_shuffle=True,
                        batch_size=FLAGS.train_batch_size)
    test_set = DataSet(data_file='/home/hongxwing/Workspace/Datas/MNIST/test_data.npy',
                       label_file='/home/hongxwing/Workspace/Datas/MNIST/test_label.npy',
                       is_shuffle=True,
                       batch_size=FLAGS.test_batch_size)
    # check_dataset(train_set)
    # check_dataset(test_set)
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    net = MNIST(FLAGS.hidden_units)
    sess = tf.Session()
    summary_writer = tf.train.SummaryWriter('.', sess.graph)
    init = tf.initialize_all_variables()
    sess.run(init)
    
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(net.loss)

    n_step = 3001
    for i in range(n_step):
        data, label = train_set.next_batch()
        # data, label = mnist.train.next_batch(FLAGS.batch_size)
        [loss_v, _] = sess.run([net.loss, train_step], feed_dict={net.inputs:data, net.label:label})
        if i%100 == 0:
            print('setp={0}, loss={1}.'.format(i, loss_v))
        if i%500 == 0:
            [accuracy_train] = sess.run([net.accuracy], feed_dict={net.inputs:data, net.label:label})
            # data_test, label_test = mnist.test.next_batch(FLAGS.batch_size)
            data_test, label_test = train_set.next_batch()
            [accuracy_test] = sess.run([net.accuracy], feed_dict={net.inputs:data_test, net.label:label_test})
            print('setp={0}, train accuracy = {1}, test accuracy = {2}.'.format(i, accuracy_train, accuracy_test))

    
def main(argv):
    test(argv)

if __name__=='__main__':
    define_flags(sys.argv)
    tf.app.run()
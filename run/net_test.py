#!/usr/bin/python
from __future__ import absolute_import, division, print_function
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import xlearn.nets.layers

from xlearn.reader.mnist import DataSet
from xlearn.model.mnist import MNISTConv
from xlearn.nets.model import NetManager

FLAGS = tf.app.flags.FLAGS

def define_flags(argv):    
    flag = tf.app.flags
    flag.DEFINE_float("weight_decay", 0.0, "Weight decay coefficient.")
    flag.DEFINE_float("eps", 1e-5, "Weight decay coefficient.")    
    flag.DEFINE_integer("train_batch_size", 100, "Batch size.")
    flag.DEFINE_integer("test_batch_size", 1000, "Batch size.")
    flag.DEFINE_integer("hidden_units", 1024, "Batch size.")
    flag.DEFINE_float("learning_rate_init", 1e-4, "Initial learning rate.")
    flag.DEFINE_string("save_dir",'.',"saving path.")
    flag.DEFINE_string("summary_dir",'.',"summary path.")
    flag.DEFINE_integer("decay_steps",100000,"decay steps.")
    flag.DEFINE_float("learning_rate_decay_factor",0.1,"learing rate decay factor.")

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

    net = MNISTConv(FLAGS.hidden_units)
    manager = NetManager(net)    

    n_step = 20001
    for i in range(n_step):
        data, label = train_set.next_batch()
        [loss_v, _] = manager.run([net.loss, net.train], feed_dict={net.inputs:data, net.label:label, net.keep_prob:0.5})        
        if i%100 == 0:
            print('setp={0}, loss={1}.'.format(i, loss_v))
        if i%500 == 0:
            [accuracy_train] = manager.run([net.accuracy], feed_dict={net.inputs:data, net.label:label, net.keep_prob:1.0})
            # data_test, label_test = mnist.test.next_batch(FLAGS.batch_size)
            data_test, label_test = train_set.next_batch()
            [accuracy_test] = manager.run([net.accuracy], feed_dict={net.inputs:data_test, net.label:label_test, net.keep_prob:1.0})
            print('setp={0}, train accuracy = {1}, test accuracy = {2}.'.format(i, accuracy_train, accuracy_test))

    
def main(argv):
    test(argv)

if __name__=='__main__':
    define_flags(sys.argv)
    xlearn.nets.model.before_net_definition()
    tf.app.run()
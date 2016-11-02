from __future__ import absolute_import, division, print_function
import sys
import os.path
import time
import random

import numpy as np
from six.moves import xrange

import tensorflow as tf

import mypylib.image as mpli
import mypylib.tensor as mplt

FLAGS = tf.app.flags.FLAGS

USE_FLICKR_DATA = False
if USE_FLICKR_DATA:
    DATA_PATH = "/home/hongxwing/Workspace/Datas/flickr25k_gray_npy/"
else:
    DATA_PATH = "/home/hongxwing/Workspace/Datas/SinogramData/"

TEST_NAME = "crop_test_deep_conv_sino"

# CROPP_FILE = False
# if CROPP_FILE:
#     pass
# else:
#     DATA_PATH = ""

BATCH_SIZE = 128
PATCH_SHAPE = [61, 61]
STRIDE = [8, 8]
CONV_STEP = 12
VALID_OFFSET = [CONV_STEP, CONV_STEP]
VALID_SHAPE = [PATCH_SHAPE[0]-2*CONV_STEP, PATCH_SHAPE[1]-2*CONV_STEP]


RESTORE = False

if USE_FLICKR_DATA:
    NTRAIN = 20000
    NTEST = 5000
else:
    NTRAIN = 200
    NTEST = 50



TRAIN_IDS = list(xrange(NTRAIN))
TEST_IDS = list(xrange(NTRAIN, NTRAIN+NTEST))
if USE_FLICKR_DATA:
    TRAIN_IDS.remove(1008)
    TRAIN_IDS.remove(5123)
    TRAIN_IDS.remove(5951)
    TRAIN_IDS.remove(9446)
    TRAIN_IDS.remove(10570)
    TRAIN_IDS.remove(13856)
    TRAIN_IDS.remove(15456)

def define_flags():
    """
    define flags for tf.app.flags
    """

    flag = tf.app.flags
    flag.DEFINE_string("data_dir", DATA_PATH, "Path to data directory")
    #flag.DEFINE_string("data_dir","/home/hongxwing/Workspace/Datas/SinogramPatchData/","Path to data directory")
    if USE_FLICKR_DATA:
        flag.DEFINE_string("prefix_h", "imh", "Prefix of high resolution data filename.")
        flag.DEFINE_string("prefix_l", "iml", "Prefix of low resoution data filename.")
    else:
        flag.DEFINE_string("prefix_h", "sinogram_high_", "Prefix of high resolution data filename.")
        flag.DEFINE_string("prefix_l", "sinogram_low_", "Prefix of low resoution data filename.")
    # flag.DEFINE_string("prefix_h","patch_high_","Prefix of high resolution data filename.")
    # flag.DEFINE_string("prefix_l","patch_low_","Prefix of low resoution data filename.")
    flag.DEFINE_string("suffix", ".npy", "Suffix of data filename.")
    flag.DEFINE_integer("batch_size", BATCH_SIZE, "Batch size of train dataset.")
    flag.DEFINE_integer("height", PATCH_SHAPE[0], "Height of images.")
    flag.DEFINE_integer("width", PATCH_SHAPE[1], "Width of images.")
    flag.DEFINE_integer("stride_v", STRIDE[0], "vertical stride step")
    flag.DEFINE_integer("stride_h", STRIDE[1], "horizontal stride step")
    flag.DEFINE_integer("valid_h", VALID_SHAPE[0], "valid output height")
    flag.DEFINE_integer("valid_w", VALID_SHAPE[1], "valid output width")
    flag.DEFINE_integer("valid_y", VALID_OFFSET[0], "valid output offset vertial")
    flag.DEFINE_integer("valid_x", VALID_OFFSET[1], "valid output offset horizontal")

    flag.DEFINE_integer("max_patch_img", 8, "valid output offset horizontal")
    flag.DEFINE_integer("conv_step", CONV_STEP, "convlution steps")
    summary_path = os.path.join(".", "summary", TEST_NAME)
    flag.DEFINE_string("summaries_dir", summary_path, "Path to summary directory.")


    flag.DEFINE_integer("conv_height", 3, "Convolution window height.")
    flag.DEFINE_integer("conv_width", 3, "Convolution window width.")
    flag.DEFINE_integer("output_conv_height", 3, "Convolution window height of output layer.")
    flag.DEFINE_integer("output_conv_width", 3, "Convolution window width of output layer.")
    flag.DEFINE_float("stddev", 2e-2, "Default std dev of weight variable.")
    flag.DEFINE_float("learning_rate_init", 1e-4, "Initial learning rate.")
    flag.DEFINE_float("learning_rate_decay_factor", 0.6, "Learning rate decay factor.")

    flag.DEFINE_integer("decay_epoch", 1, "Decay epoch.")
    flag.DEFINE_integer("decay_steps", 1000, "Decay steps.")

    flag.DEFINE_integer("max_step", 5000, "Max train steps.")
    flag.DEFINE_integer("max_test_step", 5, "Max train steps.")

define_flags()

from supernet.model import SuperNet
from supernet.input import DataSet
#from supernet_old.supernet_input import DataSet

def init():
    """
    initialization of Net, Session, Summary, SummaryWriter, Saver
    """
    net = SuperNet()
    summary = tf.merge_all_summaries()
    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver(tf.all_variables())
    summary_writer = tf.train.SummaryWriter(FLAGS.summaries_dir, sess.graph)
    sess.run(init_op)
    return net, sess, summary, summary_writer, saver

def test_net(net, sess, id_list_in):
    id_list = id_list_in[:]
    random.shuffle(id_list)
    file_high = [os.path.join(FLAGS.data_dir, FLAGS.prefix_h + str(id) + FLAGS.suffix) for id in id_list]
    file_low = [os.path.join(FLAGS.data_dir, FLAGS.prefix_l + str(id) + FLAGS.suffix) for id in id_list]


    data = DataSet(file_high, file_low,
                   [FLAGS.height, FLAGS.width],
                   [FLAGS.stride_v, FLAGS.stride_h],
                   use_random_shuffle=True,
                   max_patch_image=FLAGS.max_patch_img)
    # data = DataSet(FLAGS.data_dir, FLAGS.prefix_h, FLAGS.prefix_l, list(xrange(5,6)), FLAGS.suffix)
    loss_ave = 0.0
    # pg = data.batch_generator(FLAGS.batch_size)
    for i in xrange(FLAGS.max_test_step):
        [high_res_data, low_res_data] = data.next_batch(FLAGS.batch_size)
        # [high_res_data, low_res_data] = pg.next()
        residual = high_res_data - low_res_data
        loss_v = sess.run(net.loss, \
            feed_dict={net.low_tensor: low_res_data, net.residual: residual})
        loss_ave += loss_v
    loss_ave /= FLAGS.max_test_step
    print("[TEST]\taverge loss = ", str(loss_v))
    return loss_ave

def train_net(net, sess, summary, summary_writer, saver, id_list_in):
    id_list = id_list_in[:]
    random.shuffle(id_list)
    file_high = [os.path.join(FLAGS.data_dir, FLAGS.prefix_h + str(id) + FLAGS.suffix) for id in id_list]
    file_low = [os.path.join(FLAGS.data_dir, FLAGS.prefix_l + str(id) + FLAGS.suffix) for id in id_list]
   
    data = DataSet(file_high, file_low,
                   [FLAGS.height, FLAGS.width],
                   [FLAGS.stride_v, FLAGS.stride_h],
                   use_random_shuffle=True,
                   max_patch_image=FLAGS.max_patch_img,
                   new_crop_method=True)
    # data = DataSet(FLAGS.data_dir, FLAGS.prefix_h, FLAGS.prefix_l, list(xrange(0,4)), FLAGS.suffix)
    # pg = data.batch_generator(FLAGS.batch_size)
    for i in xrange(FLAGS.max_step):
        [high_res_data, low_res_data] = data.next_batch(FLAGS.batch_size)
        # [high_res_data, low_res_data] = pg.next()
        residual = high_res_data - low_res_data
        summary_v, loss_v, _ = sess.run([summary, net.loss, net.train_step], \
            feed_dict={net.low_tensor: low_res_data, net.residual: residual})
        print("[TRAIN]\tstep = " + str(i) + ", loss = ", str(loss_v))
        if i%10 == 0:
            summary_writer.add_summary(summary_v, i)
        if i%50 == 0 and i > 0:
            test_net(net, sess, TEST_IDS)
        if i%500 == 0 and i > 0:
            saver.save(sess, 'supernet-'+TEST_NAME, global_step=i)
    summary_writer.add_summary(summary_v, FLAGS.max_step)
    test_net(net, sess, TEST_IDS)
    saver.save(sess, 'supernet-'+TEST_NAME, global_step=FLAGS.max_step)

def infer(net, sess, image_l):
    """
    infer high resolution image using net.
    net needs to be initalized.
    """
    assert(len(image_l.shape) == 2), 'Invalid shape of image'
    tensor = np.zeros([1, image_l.shape[0], image_l.shape[1], 1])
    tensor[0, :, :, 0] = image_l

    patch_list = []
    for patch in mpli.patch_generator_tensor(tensor,
                                             [FLAGS.height, FLAGS.width],
                                             [FLAGS.stride_v, FLAGS.stride_h],
                                             None,
                                             False):
        patch_list.append(patch)

    high_resolution_list = []


    idt = FLAGS.batch_size
    while idt < len(patch_list):
        tensor = mplt.merge_patch_list(patch_list[idt-FLAGS.batch_size:idt])
        tensor_res = np.zeros(tensor.shape)
        high_resolution_image = sess.run(net.inference,
                                         feed_dict={net.low_tensor: tensor,
                                                    net.residual: tensor_res})

        for i in xrange(high_resolution_image.shape[0]):
            patch = high_resolution_image[i, :, :, 0]
            patch = np.reshape(patch, [1, FLAGS.valid_h, FLAGS.valid_w, 1])
            high_resolution_list.append(patch)
        idt += FLAGS.batch_size

    if idt > len(patch_list):
        idt -= FLAGS.batch_size
        tensor_raw = mplt.merge_patch_list(patch_list[idt:])
        tensor = np.zeros([FLAGS.batch_size, tensor_raw.shape[1], tensor_raw.shape[2], 1])
        for i in xrange(tensor_raw.shape[0]):
            tensor[i, :, :, 0] = tensor_raw[i, :, :, 0]
        tensor_res = np.zeros(tensor.shape)
        high_resolution_image = sess.run(net.inference,
                                         feed_dict={net.low_tensor: tensor,
                                                    net.residual: tensor_res})
        for i in xrange(tensor_raw.shape[0]):
            patch = high_resolution_image[i, :, :, 0]
            patch = np.reshape(patch, [1, FLAGS.valid_h, FLAGS.valid_w, 1])
            high_resolution_list.append(patch)

    high_resolution_list_correct = []
    for patch in high_resolution_list:
        patch_padding = np.zeros([1, FLAGS.height, FLAGS.width, 1])
        patch_padding[0,
                      FLAGS.valid_h:FLAGS.valid_h+FLAGS.valid_y,
                      FLAGS.valid_w:FLAGS.valid_w+FLAGS.valid_x,
                      0] = patch[0, :FLAGS.valid_y, FLAGS.valid_x, 0]
        high_resolution_list_correct.append(patch_padding)

    image_h = mpli.patches_recon_tensor(high_resolution_list,
                                        [1, image_l.shape[0], image_l.shape[1], 1],
                                        [FLAGS.height, FLAGS.width],
                                        [FLAGS.stride_v, FLAGS.stride_h],
                                        [FLAGS.valid_h, FLAGS.valid_w],
                                        # [FLAGS.valid_y, FLAGS.valid_x])
                                        [0, 0])
    return image_h

def main(argv):
    assert(len(argv) >= 2), 'Invalid command'
    net, sess, summary, summary_writer, saver = init()
    if RESTORE:
        saver.restore(sess, "./supernet-"+TEST_NAME+"-"+str(1000))
    if argv[1] == "train":
        train_net(net, sess, summary, summary_writer, saver, TRAIN_IDS)
    if argv[1] == "test":
        loss_ave = test_net(net, sess, TEST_IDS)
    if argv[1] == "infer":
        if not RESTORE:
            saver.restore(sess, "./supernet-"+TEST_NAME+"-"+str(FLAGS.max_step))
        filename = os.path.join(FLAGS.data_dir, argv[2])
        image_l = np.array(np.load(filename))
        image_h = infer(net, sess, image_l)
        np.save('result.npy', image_h)

if __name__ == "__main__":
    tf.app.run()

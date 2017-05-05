import os
import sys
import numpy as np
import tensorflow as tf
import numpy as np
from xlearn.datasets.sinogram2 import Sinograms2
from xlearn.net_tf.srmr3 import SRSino8v3
from xlearn.nets import SRInterp
import tensorflow as tf
import json
from tqdm import tqdm


def generate_samples():
    print("GENERATING Samples")
    crop_shape = get_config()['crop_shape']
    npy_file = get_config()['npy_file']
    nb_samples = get_config()['nb_samples']
    ss_input = []
    ss_label = []
    with Sinograms2(batch_size=nb_samples, is_from_npy=True, npy_file=npy_file, data_down_sample=3, crop_shape=crop_shape) as dataset:
        ss = next(dataset)
        ss_input = ss[0]
        ss_label = ss[1][2]
        np.save('ss_input.npy', ss_input)
        np.save('ss_label.npy', ss_label)
    print('ss_input_shape', ss_input.shape)


def infer_8x():
    print('INFERING 8x Samples')
    samples = np.load('ss_input.npy')
    nb_samples = get_config()['nb_samples']
    input_shape = get_config()['input_shape_8x']
    load_step = get_config()['load_step']
    period = get_config()['period_8x']
    net = SRSino8v3(filenames=['srsino8v3_8x.json'], batch_size=1,
                    input_shape=input_shape, model_dir='save8x')
    net.build()
    net.load(load_step=load_step)
    infers = []
    for i in tqdm(range(nb_samples)):
        ss = samples[i, :, :, :]
        ss = np.reshape(ss, [1] + list(ss.shape))
        infer = net.predict_fullsize(ss, period=period)
        infers.append(infer[0, :, :, :])
    infers = np.array(infers)
    print('infers8x.shape', infers.shape)
    np.save('inf8.npy', infers)
    tf.reset_default_graph()


def infer_4x():
    print('INFERING 4x Samples')
    samples = np.load('inf8.npy')
    nb_samples = get_config()['nb_samples']
    input_shape = get_config()['input_shape_4x']
    load_step = get_config()['load_step']
    period = get_config()['period_4x']
    net = SRSino8v3(filenames=['srsino8v3_4x.json'], batch_size=1,
                    input_shape=input_shape, model_dir='save4x')
    net.build()
    net.load(load_step=load_step)
    infers = []
    for i in tqdm(range(nb_samples)):
        ss = samples[i, :, :, :]
        ss = np.reshape(ss, [1] + list(ss.shape))
        infer = net.predict_fullsize(ss, period=period)
        infers.append(infer[0, :, :, :])
    infers = np.array(infers)
    print('infers4x.shape', infers.shape)
    np.save('inf4.npy', infers)
    tf.reset_default_graph()


def infer_2x():
    print('INFERING 2x Samples')
    samples = np.load('inf4.npy')
    nb_samples = get_config()['nb_samples']
    input_shape = get_config()['input_shape_2x']
    load_step = get_config()['load_step']
    period = get_config()['period_2x']
    net = SRSino8v3(filenames=['srsino8v3_2x.json'], batch_size=1,
                    input_shape=input_shape, model_dir='save2x')
    net.build()
    net.load(load_step=load_step)
    infers = []
    for i in tqdm(range(nb_samples)):
        ss = samples[i, :, :, :]
        ss = np.reshape(ss, [1] + list(ss.shape))
        infer = net.predict_fullsize(ss, period=period)
        infers.append(infer[0, :, :, :])
    infers = np.array(infers)
    print('infers2x.shape', infers.shape)
    np.save('inf2.npy', infers)
    tf.reset_default_graph()


def interpolation():
    print('INFERING interp Samples')
    samples = np.load('ss_input.npy')
    nb_samples = get_config()['nb_samples']
    input_shape = get_config()['input_shape_8x']
    full_shape = get_config()['input_shape_1x']
    ss = np.load('ss_input.npy')
    sr_ip = tf.placeholder(dtype=tf.float32, shape=[None] + input_shape)
    sr_it = tf.image.resize_images(images=sr_ip, size=full_shape[0:2])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sr_int = sess.run(sr_it, feed_dict={sr_ip: ss})
    np.save('inf_it.npy', sr_int)
    tf.reset_default_graph()


def error_analysis():
    nb_samples = get_config()['nb_samples']
    sino_l = np.load('ss_input.npy')
    sino_l = sino_l[:, :, :91, :]
    sino_h = np.load('ss_label.npy')
    sino_h = sino_h[:, :, :361, :]
    sino_sr = np.load('inf2.npy')
    sino_sr = sino_sr[:, :, :361, :]
    sino_it = np.load('inf_it.npy')
    sino_it = sino_it[:, :, :361, :]
    sino_l = np.exp(sino_l) - 1
    sino_h = np.exp(sino_h) - 1
    sino_sr = np.exp(sino_sr) - 1
    sino_it = np.exp(sino_it) - 1
    errsr = sino_h - sino_sr
    errit = sino_h - sino_it
    errsrv = []
    erritv = []
    for i in range(nb_samples):
        errsrv.append(np.sqrt(np.mean(np.square(errsr[i, ...]))))
        erritv.append(np.sqrt(np.mean(np.square(errit[i, ...]))))
    print('errsrv', errsrv)
    print('erritv', erritv)
    errsrv = np.array(errsrv)
    print('mean', np.mean(errsrv))
    print('std', np.std(errsrv))
    erritv = np.array(erritv)
    print('mean', np.mean(erritv))
    print('std', np.std(erritv))
    print('ratio', np.mean(errsrv) / np.std(erritv))
    np.save('sino_h.npy', sino_h)
    np.save('sino_l.npy', sino_l)
    np.save('sino_sr.npy', sino_sr)
    np.save('sino_it.npy', sino_it)


def check_files():
    def ckf(x): return os.path.isfile(os.path.join(os.path.abspath('.'), x))
    load_step = get_config()['load_step']
    files = []
    suffixs = ['.']
    files += [f for f in []]


def get_config():
    with open('infer.json', 'r') as fin:
        configs = json.load(fin)
    return configs


def main():
    # generate_samples()
    # infer_8x()
    # infer_4x()
    infer_2x()
    interpolation()
    error_analysis()


if __name__ == "__main__":
    main()

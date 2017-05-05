""" General entry for super resolution related tasks. """

import argparse
import datetime
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from xlearn.dataset.mnist import MNISTImage, MNIST2
from xlearn.dataset.sinogram import Sinograms
from xlearn.dataset.flickr import Flickr25k

from xlearn.nets.super_resolution import SRNetInterp, SRSimple, SRF3D, SRClassic

from xlearn.utils.general import with_config, empty_list, enter_debug
from xlearn.utils.image import subplot_images

c = dict()
c['is_cl'] = False

enter_debug()


def init():
    if c['net'] is not None:
        net = define_net()
    else:
        net = None
    if c['dataset'] is not None:
        dataset = define_dataset()
    else:
        dataset = None
    return net, dataset


def define_dataset():
    dataset_name = c['dataset']
    config_files = c['configs']
    dataset = None
    if dataset_name == 'mnist':
        dataset = MNISTImage(filenames=config_files)
    elif dataset_name == 'MNIST2':
        dataset = MNIST2(filenames=config_files)
    return dataset


def define_net():
    net_name = c['net']
    config_files = c['configs']
    net = None
    if net_name == 'interp':
        net = SRNetInterp(filenames=config_files)
    elif net_name == 'simple':
        net = SRSimple(filenames=config_files)
    elif net_name == 'classic':
        net = SRClassic(filenames=config_files)
    return net


def train(net, dataset):
    pass


def predict(net, dataset):
    pass


@with_config
def test_dataset(dataset, nb_images=64, data_type='data', settings=None, **kwargs):
    if not isinstance(data_type, (list, tuple)):
        data_type = [data_type]
    imgs_all = empty_list(len(data_type))
    for i in range(int(np.ceil(nb_images / dataset.batch_size))):
        s = next(dataset)
        for i, ctype in enumerate(data_type):
            img_tensor = dataset.data_from_sample(s, data_type=ctype)
            imgs = dataset.visualize(img_tensor)
            if imgs_all[i] is None:
                imgs_all[i] = imgs
            else:
                imgs_all[i].append(imgs)
    subplot_images(imgs_all, is_gray=True)
    for imgs in imgs_all:
        data = np.array(imgs)
        print(data.shape)
        print("mean:{0:10f}, max:{1:10f}, min:{2:10f}".format(
            np.mean(data), np.max(data), np.min(data)))


@with_config
def configure(argv, settings=None, **kwargs):
    """ read configs from console, .json file or arguments """
    parser = argparse.ArgumentParser(
        description='Super resolution general entry:')
    parser.add_argument('--run_config', '-r',
                        dest='run_config', help='run config file')
    parser.add_argument('--net', '-n', dest='net', help='net name')
    parser.add_argument('--dataset', '-d', dest='dataset', help='dataset name')
    parser.add_argument('--config', '-c', dest='configs',
                        help='config files', nargs='*')
    parser.add_argument('--task', '-t', dest='task', help='task name')
    args = parser.parse_args(argv)
    if c['is_cl'] or True:
        c['net'] = args.net
        c['dataset'] = args.dataset
        c['configs'] = args.configs
        c['task'] = args.task
    if c.get('net') is None:
        c['net'] = kwargs.get('net')

    print("=" * 50)
    print("Super resolution general entry Called.")
    print("{0:20}{1}".format("Task:", c['task']))
    print("{0:20}{1}".format("Start time:",
                             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("{0:20}{1}".format("Work dir:", os.getcwd()))
    print("{0:20}{1}".format("Net name:", c['net']))
    print("{0:20}{1}".format("Dataset name:", c['dataset']))
    print("Config files:")
    config_files = c['configs']
    if config_files is not None:
        for i, conf_file in enumerate(config_files):
            print("{0:20}{1}".format(" #%3d" % i, conf_file))
    print("=" * 50)
    print("=" * 50)
    print("{0:20}{1}".format("End time:",
                             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


def work():
    net, dataset = init()
    task = c['task']
    if task == 'train':
        train(net, dataset)
    elif task == 'predict':
        predict(net, dataset)
    elif task == 'test_dataset':
        test_dataset(dataset, filenames=c.get('config_test_dataset'))
    elif task == 'test':
        print('TEST')

if __name__ == "__main__":
    c['is_cl'] = True
    configure(sys.argv[1:])
    work()

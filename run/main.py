""" General entry for super resolution related tasks. """
import matplotlib
matplotlib.use('agg')
import argparse
import datetime
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import fire

from keras.callbacks import ModelCheckpoint

from xlearn.dataset.mnist import MNIST, MNISTImage, MNIST2
from xlearn.dataset.sinogram import Sinograms
from xlearn.dataset.flickr import Flickr25k

from xlearn.nets.super_resolution import SRNetInterp, SRSimple, SRF3D, SRClassic
from xlearn.knet.ae1d import AE1D, VAE1D, CVAE1D
from xlearn.utils.general import with_config, empty_list, enter_debug
from xlearn.utils.image import subplot_images


class DLRun:
    net = None
    dataset = None

    def __init__(self):
        self._is_debug = False

    def define_dataset(self, dataset_name, config_files=None, **kwargs):
        if self._is_debug:
            enter_debug()
        dataset = None
        if dataset_name == 'mnist':
            dataset = MNIST(filenames=config_files, **kwargs)
        elif dataset_name == 'MNIST2':
            dataset = MNIST2(filenames=config_files, **kwargs)
        if dataset is None:
            raise ValueError(
                'Unknown dataset_name {0:s}.'.format(dataset_name))
        print(dataset.pretty_settings())
        return dataset

    def define_net(self, net_name, config_files=None):
        if self._is_debug:
            enter_debug()
        net = None
        if net_name == 'AE1D':
            net = AE1D(filenames=config_files)
        elif net_name == 'VAE1D':
            net = VAE1D(filenames=config_files)
        elif net_name == 'CVAE1D':
            net = CVAE1D(filenames=config_files)
        elif net_name == 'sr_interp':
            net = SRNetInterp(filenames=config_files)
        elif net_name == 'sr_simple':
            net = SRSimple(filenames=config_files)
        elif net_name == 'sr_classic':
            net = SRClassic(filenames=config_files)
        if net is None:
            raise ValueError('Unknown net_name {0:s}.'.format(net_name))
        print(net.pretty_settings())
        net.define_net()
        return net

    @with_config
    def train(self,
              net_name=None,
              dataset_name=None,
              is_force_load=False,
              config_files=None,
              is_debug=False,
              is_reset_lr=False,
              lrs=None,
              settings=None,
              filenames=None,
              **kwargs):
        net_name = settings.get('net_name', net_name)
        dataset_name = settings.get('dataset_name', dataset_name)
        is_debug = settings.get('is_debug', is_debug)
        config_files = settings.get('config_files', config_files)
        is_reset_lr = settings.get('is_reset_lr', is_reset_lr)
        is_force_load = settings.get('is_force_load', is_force_load)
        lrs = settings.get('lrs', lrs)
        print("=" * 30)
        print("Train with setttings")
        print(settings)
        print("=" * 30)
        if is_debug:
            enter_debug()
        net = self.define_net(net_name, config_files=config_files)
        dataset = self.define_dataset(dataset_name, config_files=config_files)
        if is_force_load:
            net.load(is_force=True)
        if is_reset_lr:
            net.reset_lr(lrs)
        if net_name == 'AE1D' or 'VAE1D' or 'CVAE1D':
            net.model_ae.fit_generator(
                dataset, steps_per_epoch=1875, epochs=40, verbose=1,
                callbacks=[ModelCheckpoint(
                    filepath=r"weightsP0.{epoch:02d}-{loss:.5f}.hdf5", period=1)]
            )
            net.reset_lr([1e-4])
            net.model_ae.fit_generator(
                dataset, steps_per_epoch=1875, epochs=40, verbose=1,
                callbacks=[ModelCheckpoint(
                    filepath=r"weightsP1.{epoch:02d}-{loss:.5f}.hdf5", period=1)]
            )
            net.reset_lr([1e-5])
            net.model_ae.fit_generator(
                dataset, steps_per_epoch=1875, epochs=40, verbose=1,
                callbacks=[ModelCheckpoint(
                    filepath=r"weightsP2.{epoch:02d}-{loss:.5f}.hdf5", period=1)]
            )
        net.save()

    @with_config
    def show_mainfold(self,
                      net_name=None,
                      dataset_name=None,
                      config_files=None,
                      save_filename='./mainfold.png'):
        pass

    @with_config
    def generate(self,
                 net_name=None,
                 dataset_name=None,
                 is_debug=False,
                 config_files=None,
                 path_save='./generate',
                 is_visualize=True,
                 is_save=True,
                 save_filename='generate.png',
                 settings=None,
                 **kwargs):
        print('Genererate routine is called.')
        net_name = settings.get('net_name', net_name)
        dataset_name = settings.get('dataset_name', dataset_name)
        is_debug = settings.get('is_debug', is_debug)
        config_files = settings.get('config_files', config_files)
        path_save = settings.get('path_save', path_save)
        is_visualize = settings.get('is_visualize', is_visualize)
        is_save = settings.get('is_save', is_save)
        save_filename = settings.get('save_filename', save_filename)
        if is_debug:
            enter_debug()
        net = self.define_net(net_name, config_files=config_files)
        dataset = self.define_dataset(dataset_name, config_files=config_files)
        net.load(is_force=True)
        if net_name == 'CVAE1D':
            s = next(dataset)
            z = net.gen_noise()
            p = net.model('gen').predict(
                [z, s[0][1]], batch_size=net.batch_size)
            x = dataset.visualize(s[0][1], data_type='label')
            ae = dataset.visualize(p, data_type='data')
            subplot_images((x[:64], ae[:64]), is_gray=True,
                           is_save=True, filename=save_filename)
            np.save('generate_condition.npy', s[0][1])
            np.save('generate_input.npy', z)
            np.save('generate_output.npy', p)

    @with_config
    def predict(self,
                net_name=None,
                dataset_name=None,
                is_debug=False,
                config_files=None,
                path_save='./predict',
                is_visualize=False,
                is_save=False,
                save_filename='predict.png',
                settings=None,
                filenames=None,
                **kwargs):
        print("Predict routine is called.")
        net_name = settings.get('net_name', net_name)
        dataset_name = settings.get('dataset_name', dataset_name)
        is_debug = settings.get('is_debug', is_debug)
        config_files = settings.get('config_files', config_files)
        path_save = settings.get('path_save', path_save)
        is_visualize = settings.get('is_visualize', is_visualize)
        is_save = settings.get('is_save', is_save)
        save_filename = settings.get('save_filename', save_filename)
        if is_debug:
            enter_debug()
        net = self.define_net(net_name, config_files=config_files)
        dataset = self.define_dataset(dataset_name, config_files=config_files)
        net.load(is_force=True)
        if net_name == 'AE1D' or net_name == 'VAE1D':
            s = next(dataset)
            p = net.model('ae').predict(s[0], batch_size=net.batch_size)
            if dataset_name == 'MNIST2':
                x = dataset.visualize(s[0], data_type='label')
            else:
                x = dataset.visualize(s[0])
            ae = dataset.visualize(p)
            subplot_images((x[:64], ae[:64]), is_gray=True,
                           is_save=True, filename=save_filename)
            np.save('predict_input.npy', s[0])
            np.save('predict_output.npy', p)
        if net_name == 'CVAE1D':
            s = next(dataset)
            p = net.model('ae').predict(s[0], batch_size=net.batch_size)

            xs = dataset.visualize(s[0][0], data_type='data')
            xl = dataset.visualize(s[0][1], data_type='label')
            ae = dataset.visualize(p, data_type='data')
            subplot_images((xs[:64], xl[:64], ae[:64]), is_gray=True,
                           is_save=True, filename=save_filename)
            np.save('predict_data.npy', s[0][0])
            np.save('predict_label.npy', s[0][1])
            np.save('predict_output.npy', p)

    @with_config
    def test_dataset(self, dataset_name, config_files=None, nb_images=64, data_type='data', img_filename='result.png', is_save=False,  filenames=None, settings=None, **kwargs):
        if self._is_debug:
            enter_debug()
        config_files = settings.get('config_files', config_files)
        nb_images = settings.get('nb_images', nb_images)
        data_type = settings.get('data_type', data_type)
        img_filename = settings.get('img_filename', img_filename)
        is_save = settings.get('is_save', is_save)
        if not isinstance(data_type, (list, tuple)):
            data_type = [data_type]
        imgs_all = empty_list(len(data_type))
        dataset = self.define_dataset(dataset_name, config_files=config_files)
        for i in range(int(np.ceil(nb_images / dataset.batch_size))):
            s = next(dataset)
            for i, ctype in enumerate(data_type):
                img_tensor = dataset.data_from_sample(s, data_type=ctype)
                imgs = dataset.visualize(img_tensor)
                if imgs_all[i] is None:
                    imgs_all[i] = imgs
                else:
                    imgs_all[i].append(imgs)
        subplot_images(imgs_all, is_gray=True,
                       is_save=is_save, filename=img_filename)
        # for imgs in imgs_all:
        #     data = np.array(imgs)
        #     print(data.shape)
        #     print("mean:{0:10f}, max:{1:10f}, min:{2:10f}".format(
        #         np.mean(data), np.max(data), np.min(data)))

    def test_run(self, **kwargs):
        print('Test run called.')
        print(kwargs)


if __name__ == "__main__":
    fire.Fire(DLRun)

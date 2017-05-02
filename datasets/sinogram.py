""" standard sinogram dataset class(es) """

import os
import random
import h5py
import numpy as np
from ..utils.general import with_config
from ..utils.cells import Sampler
from ..utils.prints import pp_json


class Sinograms:
    """ Sinogram dataset """
    @with_config
    def __init__(self,
                 batch_size,
                 crop_shape,
                 mode='train',
                 data_down_sample=3,
                 label_down_sample=0,
                 padding=3,
                 period=360,
                 sino_name='shepplogan',
                 is_from_npy=False,
                 npy_file='sino.npy',
                 is_norm=False,
                 data_mean=6.0,
                 data_std=0.5,
                 offset=(0, 0),
                 **kwargs):
        """ init of Sinograms """
        dataset_dir = os.environ.get('PATH_DATASETS')
        self.is_from_npy = is_from_npy
        if not self.is_from_npy:
            self.file_data = os.path.join(
                dataset_dir, sino_name + '_sinograms.h5')
        else:
            self.file_data = npy_file
        self.dataset = None
        self.batch_size = batch_size
        self.crop_shape = crop_shape
        self.data_down_sample = data_down_sample
        self.label_down_sample = label_down_sample
        self.npy_file = npy_file
        self.padding = padding
        self.period = period
        self.is_norm = is_norm
        self.data_mean = data_mean
        self.data_std = data_std
        self.mode = mode
        self.offset = offset

    def _load_sample(self):
        id_ = next(self.sampler)[0]
        if self.is_from_npy:
            image = np.array(self.dataset[id_], dtype=np.float32)
        else:
            image = np.array(self.dataset[id_], dtype=np.float32)
        image = image[:, :self.period, :]
        image = [image] * self.padding
        image = np.concatenate(image, axis=1)
        return image

    def init(self):
        if self.is_from_npy:
            self.dataset = np.load(self.file_data)
            self.idx = list(range(self.dataset.shape[0]))
            self.sampler = Sampler(self.idx, is_shuffle=False)
        else:
            self.fin = h5py.File(self.file_data, 'r')
            self.dataset = self.fin['sinograms']
            nb_dataset_samples = self.dataset.shape[0]
            nb_train = int(nb_dataset_samples * 0.8)
            if self.mode == 'train':
                self.idx = list(range(nb_train))
            elif self.mode == 'test':
                self.idx = list(range(nb_train, nb_dataset_samples))
            else:
                raise ValueError('Invalid mode name %s.' % self.mode)
            self.sampler = Sampler(self.idx, is_shuffle=True)
        cfg_dict = {
            'batch_size': self.batch_size,
            'crop_shape': self.crop_shape,
            'mode': self.mode,
            'data_down_sample': self.data_down_sample,
            'label_down_sample': self.label_down_sample,
            'padding': self.padding,
            'period': self.period,
            'file_data': self.file_data,
            'is_norm': self.is_norm,
            'data_mean': self.data_mean,
            'data_std': self.data_std,
            'offset': self.offset,
            'dataset_shape': self.dataset.shape
        }
        pp_json(cfg_dict, 'CONFIGS OF SINOGRAM DATASET:')

    def close(self):
        if not self.is_from_npy:
            self.fin.close()

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, etype, value, traceback):
        self.close()
        # raise etype(value, traceback)

    def _crop(self, image_ip):
        """ crop image into small patch """
        image = np.array(image_ip)
        target_shape = self.crop_shape
        offsets = list(self.offset)
        is_crop_random = True

        if is_crop_random:
            if target_shape[0] < image.shape[0]:
                offsets[0] += random.randint(0,
                                             image.shape[0] - target_shape[0] - 1)
            if target_shape[1] < image.shape[1]:
                offsets[1] += random.randint(0,
                                             image.shape[1] - target_shape[1] - 1)
        image = image[offsets[0]:offsets[0] + target_shape[0],
                      offsets[1]:offsets[1] + target_shape[1], :]
        if image.shape[0] != self.crop_shape[0]:
            raise ValueError('Wrong shape, image.shape {0}, targe_shape {1}'.format(
                image.shape, self.crop_shape))
        if image.shape[1] != self.crop_shape[1]:
            raise ValueError('Wrong shape, image.shape {0}, targe_shape {1}'.format(
                image.shape, self.crop_shape))
        return image

    def sample_single(self):
        img = self._load_sample()
        img_c = self._crop(img)
        max_down_sample = max(self.data_down_sample, self.label_down_sample)
        imgs_label = []
        imgs_data = []
        imgs_data.append(img_c)
        img = img_c

        # Down sample with mean
        for i in range(max_down_sample):
            imgl = img[:, ::2, :]
            imgr = img[:, 1::2, :]
            imgs_label.append([imgl, imgr, img])
            img = (imgl + imgr) / 2.0
            imgs_data.append(img)

        data = imgs_data[self.data_down_sample]
        label = imgs_label[self.label_down_sample]
        data = np.array(data)
        label[0] = np.array(label[0])
        label[1] = np.array(label[1])
        label[2] = np.array(label[2])
        if self.is_norm:
            data -= self.data_mean
            data /= self.data_std
        for i in range(3):
            if self.is_norm:
                label[i] -= self.data_mean
                label[i] /= self.data_std        
        return data, label

    def sample(self):
        data = []
        labell = []
        labelr = []
        labelf = []
        for i in range(self.batch_size):
            ss = self.sample_single()
            data.append(ss[0])
            labell.append(ss[1][0])
            labelr.append(ss[1][1])
            labelf.append(ss[1][2])
        data = np.array(data)
        labell = np.array(labell)
        labelr = np.array(labelr)
        labelf = np.array(labelf)
        label = (labell, labelr, labelf)
        return data, label

    def __next__(self):
        return self.sample()

    def vis(self, data):
        de_norm_data = data + self.data_std
        de_norm_data *= self.data_mean
        de_norm_data = de_norm_data[:, :, :, 0]
        de_norm_data = list(de_norm_data)
        return de_norm_data

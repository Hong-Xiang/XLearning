import numpy as np
import h5py
import os
import random
from ..utils.general import with_config
from ..utils.cells import Sampler


class SinogramsPETRebin:
    @with_config
    def __init__(self,
                 batch_size,
                 crop_shape,
                 mode='train',
                 data_down_sample=3,
                 label_down_sample=0,
                 is_from_npy=False,
                 npy_file='sino.npy',
                 **kwargs):
        dataset_dir = os.environ.get('PATH_DATASETS')
        self.file_data = os.path.join(dataset_dir, 'pet_rebin.h5')
        with h5py.File(self.file_data, 'r') as fin:
            dataset = fin['images']
            nb_dataset_samples = dataset.shape[0]
        nb_train = int(nb_dataset_samples * 0.8)
        if mode == 'train':
            self.idx = list(range(nb_train))
        elif mode == 'test':
            self.idx = list(range(nb_train, nb_dataset_samples))
        else:
            raise ValueError('Invalid mode name %s.' % mode)
        self.sampler = Sampler(self.idx, is_shuffle=True)
        self.dataset = None
        self.batch_size = batch_size
        self.crop_shape = crop_shape
        self.data_down_sample = data_down_sample
        self.label_down_sample = label_down_sample
        self.is_from_npy = is_from_npy
        self.npy_file = npy_file
        if self.is_from_npy:
            tmp = np.array(np.load(self.npy_file))
            self.idx = list(range(tmp.shape[0]))
            self.sampler = Sampler(self.idx, is_shuffle=False)

    def _load_sample(self):
        if self.is_from_npy:
            image = np.load(self.npy_file)
        else:
            valid = False
            while not valid:
                id_ = next(self.sampler)[0]
                slice_id = id_ % 159
                if slice_id <= 130:
                    valid = True
            image = np.array(self.dataset[id_])
        image = image.T
        image = np.reshape(image, [image.shape[0], image.shape[1], 1])
        image = np.concatenate((image, image), axis=1)
        image += 11.0
        image = np.log(image)
        image /= 5.0
        image -= 0.5
        return image

    def init(self):
        if self.is_from_npy:
            self.dataset = np.load(self.npy_file)
        else:
            self.fin = h5py.File(self.file_data, 'r')
            self.dataset = self.fin['images']

    def close(self):
        if not self.is_from_npy:
            self.fin.close()

    def __enter__(self):
        self.fin = h5py.File(self.file_data, 'r')
        self.dataset = self.fin['images']
        return self

    def __exit__(self, etype, value, traceback):
        self.fin.close()
        # raise etype(value, traceback)

    def _crop(self, image_ip):
        """ crop image into small patch """
        image = np.array(image_ip)
        target_shape = self.crop_shape
        crop_offset = [0, 0]
        offsets = crop_offset
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
        for i in range(max_down_sample):
            imgl = img[:, ::2, :]
            imgr = img[:, 1::2, :]
            imgs_label.append((imgl, imgr, img))
            img = (imgl + imgr) / 2.0
            imgs_data.append(img)
        data = imgs_data[self.data_down_sample]
        label = imgs_label[self.label_down_sample]
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

    def vis(self, data, mode='data'):
        if mode == 'data':
            data = np.array(data[:, :, :, 0], dtype=np.float32)
            return list(data)
        if mode == 'labels':
            datal = np.array(data[0][:, :, :, 0], dtype=np.float32)
            datar = np.array(data[1][:, :, :, 0], dtype=np.float32)
            return (list(datal), list(datar))
        if mode == 'merge':
            shape = list(data[0].shape)
            shape[2] *= 2
            data_merged = np.zeros(shape=shape)
            data_merged[:, :, ::2, :] = data[0]
            data_merged[:, :, 1::2, :] = data[1]
            data_merged = data_merged[:, :, :, 0]
            return list(data_merged)

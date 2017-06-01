import numpy as np
from .base import DataSetImages
from ..utils.general import with_config


class SinoPhan(DataSetImages):
    @with_config
    def __init__(self,
                 nb_evts=1e7,
                 is_poi=False,
                 dataset_name='sino_phan',
                 **kwargs):
        DataSetImages.__init__(self, dataset_name='sino_phan', **kwargs)
        nb_padding = int(np.ceil(self.p.crop_shape[1] / 320))
        self.params['padding'] = [1, nb_padding]
        self.params['nb_evts'] = nb_evts
        self.params['is_poi'] = is_poi
        self.params.update_short_cut()
        self.params['keys'] = ['data', 'label', 'idx']
        if self.params['is_down_sample']:
            for i in range(self.p.nb_down_sample + 1):
                self.params['keys'].append('data%d' % i)
                self.params['keys'].append('label%d' % i)
        self.params.update_short_cut()

    def _load_sample(self):
        failed = True
        nb_failed = 0
        while failed:
            failed = False
            idx = next(self.sampler)[0]
            image = np.array(self.dataset[idx], dtype=np.float32)
            image = np.abs(image)
            mean_value = np.mean(image)
            std_value = np.std(image)
            if len(image.shape) == 2:
                image = image.reshape([image.shape[0], image.shape[1]])
            elif len(image.shape) == 3:
                image = np.mean(image, axis=2)
            if self.p.padding is not None:
                image = image[:self.p.period[0], :self.p.period[1]]
                # images = [image] * (self.p.padding[0] + 1)
                # image = np.concatenate(images, axis=0)
                images = [image] * (self.p.padding[1] + 1)
                image = np.concatenate(images, axis=1)
            if image.shape[0] < self.p.crop_offset[0] + self.p.crop_shape[0]:
                failed = True
            if image.shape[1] < self.p.crop_offset[1] + self.p.crop_shape[1]:
                failed = True
            if not failed:
                image = self.crop(image)
            nnz = len(np.nonzero(image > 1e-5)[0])
            rnz = nnz / np.size(image)
            if rnz < self.p.nnz_ratio:
                failed = True
            nb_failed += 1
            if nb_failed > 100:
                raise ValueError('Tried load more than 100 images and failed.')
        if self.p.is_poi:
            image, noise_image, mean_value, std_value = self._add_noise(image)
        return image, noise_image, idx, mean_value, std_value

    def _add_noise(self, image):
        image = np.array(image)
        nb_evts = np.sum(image)
        image = image / nb_evts * self.p.nb_evts
        noise_image = np.random.poisson(image, size=image.shape)
        mean_value = np.mean(noise_image)
        std_value = np.std(noise_image)
        return image, noise_image, mean_value, std_value

    def _sample_single(self):
        """ read from dataset HDF5 file and perform necessary preprocessings """
        image, noise_image, idx, mean_value, std_value = self._load_sample()

        if self.p.is_gray:
            image = np.mean(image, axis=0, keepdims=True)

        if self.p.is_down_sample:
            data_clean = []
            data_noise = []
            data_clean.append(np.array(image))
            data_noise.append(np.array(noise_image))
            for i in range(self.p.nb_down_sample):
                image = self.downsample(image)
                noise_image = self.downsample(noise_image)
                data_clean.append(image)
                data_noise.append(noise_image)

            out = {
                'data': data_noise[self.p.data_down_sample],
                'label': data_clean[self.p.label_down_sample],
                'idx': idx
            }
            for i in range(self.p.nb_down_sample + 1):
                out.update({'data%d' % i: data_noise[i]})
                out.update({'label%d' % i: data_clean[i]})
        else:
            out = {'label': image, 'data': noise_image, 'idx': idx}
            down_sampled = np.reshape(image, list(image.shape) + [1])

        for k in out:
            if k == 'idx':
                continue
            if self.p.is_norm:
                out[k] = self.norm(out[k], mean_value, std_value)
            out[k] = self.padding_channel(out[k])
        return out

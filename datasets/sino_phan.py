import np as np
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
        nb_padding = int(np.ceil(self.p.crop_shape[1]/320))
        self.params['padding'] = [1, nb_padding]
        self.params['nb_evts'] = nb_evts
        self.params['is_poi'] = is_poi
        self.params.update_short_cut()
    
    def _load_sample(self):
        failed = True
        nb_failed = 0
        while failed:
            failed = False
            idx = next(self.sampler)[0]
            image = np.array(self.dataset[idx], dtype=np.float32)
            mean_value = np.mean(image)
            std_value = np.std(image)
            if len(image.shape) == 2:
                image = image.reshape([image.shape[0], image.shape[1]])
            elif len(image.shape) == 3:
                image = np.mean(image, axis=2)
            if self.p.padding is not None:
                image = image[:self.p.period[0], :self.p.period[1]]
                images = [image] * (self.p.padding[0] + 1)
                image = np.concatenate(images, axis=0)
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
        return image, idx, mean_value, std_value
    
    def _add_noise(self, image):
        image = np.array(image)
        nb_evts = np.sum(images)
        image = image / nb_evts * self.p.nb_evts
        image = np.random.poission(image, size=image.shape)
        mean_value = np.mean(image)
        std_value = np.std(image)
        return image, mean_value, std_value
    def _sample_single(self):
        """ read from dataset HDF5 file and perform necessary preprocessings """
        image, idx, mean_value, std_value = self._load_sample()

        if self.p.is_gray:
            image = np.mean(image, axis=0, keepdims=True)
        
        if self.p.is_poi:
            image, mean_value, std_value = self._add_noise(image)
        
        if self.p.is_down_sample:
            down_sampled = []
            down_sampled.append(np.array(image))
            for i in range(self.p.nb_down_sample):                        
                image = self.downsample(image)
                down_sampled.append(np.array(image))

            out = {
                'data': down_sampled[self.p.data_down_sample],
                'label': down_sampled[self.p.label_down_sample],
                'idx': idx
            }
            for i in range(self.p.nb_down_sample+1):
                out.update({'data%d'%i: down_sampled[i]})
        else:
            out = {'data': image, 'idx': idx}
            down_sampled = np.reshape(image, list(image.shape)+[1])
        
            
        for k in out:
            if k == 'idx':
                continue            
            if self.p.is_norm:
                out[k] = self.norm(out[k], mean_value, std_value)
            out[k] = self.padding_channel(out[k])        
        return out
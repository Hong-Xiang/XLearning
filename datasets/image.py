# IMAGE_CONFIG_FILE = 'image_datasets.json'


# class DatasetImages(DatasetBase):
#     """ base class for image datasets, thus
#     next(dataset)['data'] should has format NHWC
#     In default:
#         raw data is assumed in range [0, M)
#     """
#     @json_config
#     @auto_configs
#     def __init__(self, *,
#                  raw_file_name=None,
#                  raw_type='hdf5',
#                  raw_keys=None,
#                  mode='train',
#                  is_with_idx=None,
#                  idx_file_name=None,
#                  batch_size=32,
#                  is_norm=False,
#                  norm_type='share',
#                  norm_gamma=1.0,
#                  norm_mean=0.0,
#                  norm_std=1.0,
#                  is_tf=False,
#                  queue_threads=None,
#                  queue_capacity=None,
#                  is_gray=False,
#                  is_uint8=False,
#                  crop_shape=None,
#                  crop_offset=(0, 0),
#                  is_crop_random=True,
#                  is_down_sample=False,
#                  data_down_sample=3,
#                  label_down_sample=0,
#                  nb_down_sample=3,
#                  is_down_sample_dims=True,
#                  down_sample_method='mean',
#                  nnz_ratio=0.0,
#                  padding=None,
#                  period=None,
#                  data_format='channels_last',
#                  name='DatasetImage'):
#         """
#         Inputs:
#         is_using_default: use default infos, override other settings.
#         dataset_name: load configs from $PATH_XLEARN/configs/dataset/image_datasets.json
#         """
#         base_args = {
#             'raw_file_name': raw_file_name,
#             'raw_keys': raw_keys,
#             'raw_type': raw_type,
#             'is_width_idx': is_with_idx
#         }
#         raw_file_name=None,
#                  raw_type='hdf5',
#                  raw_keys=None,
#                  is_full_load=False,
#                  mode='train',
#                  is_with_idx=None,
#                  idx_file_name=None,
#                  batch_size=32,
#                  is_norm=False,
#                  norm_type='share',
#                  norm_gamma=1.0,
#                  norm_mean=0.0,
#                  norm_std=1.0,
#                  is_tf=False,
#                  queue_threads=None,
#                  queue_capacity=None,
#                  name='dataset'
#         super(DataSetImages, self).__init__(**kwargs)
#         self.params['dataset_name'] = dataset_name
#         self.params['is_using_default'] = is_using_default

#         self.params['is_full_load'] = is_full_load
#         self.params['is_gray'] = is_gray
#         self.params['is_uint8'] = is_uint8

#         self.params['is_down_sample'] = is_down_sample
#         self.params['is_down_sample_0'] = is_down_sample_0
#         self.params['is_down_sample_1'] = is_down_sample_1
#         self.params['data_down_sample'] = data_down_sample
#         self.params['label_down_sample'] = label_down_sample
#         self.params['nb_down_sample'] = nb_down_sample
#         self.params['down_sample_method'] = down_sample_method

#         self.params['crop_shape'] = crop_shape
#         self.params['crop_offset'] = crop_offset
#         self.params['is_crop_random'] = is_crop_random

#         self.params['padding'] = padding
#         self.params['period'] = period
#         self.params['nnz_ratio'] = nnz_ratio

#         self.params['data_format'] = data_format

#         if self.params['is_using_default']:
#             self.params.update_short_cut()
#             self.load_default_json()
#         self.params.update_short_cut()
#         if self.params['is_uint8']:
#             self.params['data_mean'] = 128
#             self.params['data_std'] = 128

#         if self.params['is_down_sample']:
#             down_sample_ratio = [1, 1]
#             if self.p.is_down_sample_0:
#                 down_sample_ratio[0] = 2
#             if self.p.is_down_sample_1:
#                 down_sample_ratio[1] = 2
#             self.params['down_sample_ratio'] = down_sample_ratio

#         if isinstance(self.params['data_key'], str):
#             data_key = self.params.get('data_key', data_key)
#             self.params['data_key'] = {self.params['mode']: data_key}

#         if self.p.is_down_sample:
#             nb_down = self.params['nb_down_sample']
#             self.params['keys'] = ['data%d'%n for n in range(nb_down+1)] + ['data', 'label', 'idx']
#         else:
#             self.params['keys'] = ['data', 'idx']

#         self.params.update_short_cut()
#         self.dataset = None
#         self.fin = None


#     def load_default_json(self):
#         p = pathlib.Path(PATH_XLEARN) / 'configs' / 'dataset' / IMAGE_CONFIG_FILE
#         with open(str(p.absolute()), 'r') as fin:
#             params_default = json.load(fin)[self.p.dataset_name]
#         self.params.update(params_default)


#     def crop(self, image_ip):
#         """ crop *ONE* image into small patch """
#         image = numpy.array(image_ip, dtype=numpy.float32)
#         target_shape = list(self.p.crop_shape)
#         # if self.p.is_down_sample:
#         #     target_shape[0] *= (self.p.down_sample_ratio[0] ** self.p.nb_down_sample)
#         #     target_shape[1] *= (self.p.down_sample_ratio[1] ** self.p.nb_down_sample)

#         offsets = list(self.p.crop_offset)
#         is_crop_random = self.p.is_crop_random
#         if is_crop_random:
#             max0 = image.shape[0] - target_shape[0] - 1
#             if max0 > 0:
#                 offsets[0] += self.rand_int(minv=0, maxv=max0)
#             else:
#                 offsets[0] = 0
#             max1 = image.shape[1] - target_shape[1] - 1
#             if max1 > 0:
#                 offsets[1] += self.rand_int(minv=0, maxv=max1)
#             else:
#                 offsets[1] = 0
#         if offsets[0] + target_shape[0] > image.shape[0]:
#             raise ValueError('Too large crop shape or offset with image.shape[0]=%d, offset[0]+target_shape[0]=%d.'%(image.shape[0], offsets[0] + target_shape[0]))
#         if offsets[1] + target_shape[1] > image.shape[1]:
#             raise ValueError('Too large crop shape or offset with image.shape[1]=%d, offset[1]+target_shape[1]=%d.'%(image.shape[1], offsets[1] + target_shape[1]))

#         image = image[offsets[0]:offsets[0] + target_shape[0],
#                       offsets[1]:offsets[1] + target_shape[1]]
#         return image

#     def downsample(self, image):
#         """ down sample *ONE* image/patch """
#         image = numpy.array(image, dtype=numpy.float32)
#         image_d = downsample(image, self.p.down_sample_ratio, method=self.p.down_sample_method)
#         return image_d

#     def visualize(self, sample, is_no_change=False):
#         """ convert numpy.ndarray data into list of plotable images """
#         # Decouple visualization of data
#         images = numpy.array(sample, dtype=numpy.float32)

#         # Remove axis 1 for gray images.
#         if sample.shape[1] == 1:
#             images = images.reshape(
#                 [images.shape[0], images.shape[2], images.shape[3]])
#         else:
#             images = images

#         # Inverse normalization
#         if not is_no_change:
#             images = self.denorm(images)

#         if self.is_uint8 and not is_no_change:
#             images = numpy.array(images, dtype=numpy.uint8)
#         images = list(images)
#         return images

#     def initialize(self):
#         super(DataSetImages, self).initialize()
#         p = pathlib.Path(PATH_DATASETS)/self.p.file_data
#         if self.p.file_type == 'npy':
#             self.fin = str(p.absolute())
#             self.dataset = numpy.load(self.fin)
#         elif self.p.file_type == 'h5':
#             self.fin = h5py.File(str(p.absolute()), 'r')
#             if self.p.is_full_load:
#                 self.dataset = numpy.array(self.fin[self.p.data_key[self.p.mode]])
#             else:
#                 self.dataset = self.fin[self.p.data_key[self.p.mode]]
#         else:
#             raise TypeError('Invaid file type {}.'.format(self.p.file_type))
#         if len(self.p.data_key.keys()) > 1:
#             self.nb_examples = self.dataset.shape[0]
#             self.sampler = Sampler(
#                 list(range(self.nb_examples)), is_shuffle=True)
#         else:
#             nb_data = self.dataset.shape[0]
#             nb_train = nb_data // 5 * 4
#             if self.p.is_fix_idx:
#                 idxs = numpy.load('idx_'+self.p.mode+'.npy')
#                 self.sampler = Sampler(idxs, is_shuffle=True)
#                 self.nb_examples = len(idxs)
#             else:
#                 if self.p.mode == 'train':
#                     self.nb_examples = nb_train
#                     self.sampler = Sampler(
#                         list(range(self.nb_examples)), is_shuffle=True)
#                 else:
#                     self.nb_examples = nb_data - nb_train
#                     self.sampler = Sampler(
#                         list(range(nb_train, nb_data)), is_shuffle=True)

#     def finalize(self):
#         super(DataSetImages, self).finalize()
#         if self.p.file_type == 'h5':
#             self.fin.close()

#     def _load_sample(self):
#         failed = True
#         nb_failed = 0
#         while failed:
#             failed = False
#             idx = next(self.sampler)[0]
#             image = numpy.array(self.dataset[idx], dtype=numpy.float32)

#             if len(image.shape) == 2:
#                 image = image.reshape([image.shape[0], image.shape[1]])
#             elif len(image.shape) == 3:
#                 image = numpy.mean(image, axis=2)
#             if self.p.padding is not None:
#                 image = image[:self.p.period[0], :self.p.period[1]]
#                 images = [image] * (self.p.padding[0] + 1)
#                 image = numpy.concatenate(images, axis=0)
#                 images = [image] * (self.p.padding[1] + 1)
#                 image = numpy.concatenate(images, axis=1)
#             if image.shape[0] < self.p.crop_offset[0] + self.p.crop_shape[0]:
#                 failed = True
#             if image.shape[1] < self.p.crop_offset[1] + self.p.crop_shape[1]:
#                 failed = True
#             if not failed:
#                 image = self.crop(image)
#             nnz = len(numpy.nonzero(image > 1e-5)[0])
#             rnz = nnz / numpy.size(image)
#             if rnz < self.p.nnz_ratio:
#                 failed = True
#             nb_failed += 1
#             if nb_failed > 100:
#                 raise ValueError('Tried load more than 100 images and failed.')
#         return image, idx

#     def padding_channel(self, image):
#         if len(image.shape) == 2:
#             image = image.reshape(list(image.shape)+[1])
#         return image

#     def _sample_single(self):
#         """ read from dataset HDF5 file and perform necessary preprocessings """
#         image, idx = self._load_sample()

#         if self.p.is_gray:
#             image = numpy.mean(image, axis=0, keepdims=True)


#         if self.p.is_down_sample:
#             down_sampled = []
#             down_sampled.append(numpy.array(image))
#             for i in range(self.p.nb_down_sample):
#                 image = self.downsample(image)
#                 down_sampled.append(numpy.array(image))

#             out = {
#                 'data': down_sampled[self.p.data_down_sample],
#                 'label': down_sampled[self.p.label_down_sample],
#                 'idx': idx
#             }
#             for i in range(self.p.nb_down_sample+1):
#                 out.update({'data%d'%i: down_sampled[i]})
#         else:
#             out = {'data': image, 'idx': idx}
#             down_sampled = numpy.reshape(image, list(image.shape)+[1])


#         for k in out:
#             if k == 'idx':
#                 continue
#             if self.p.is_norm:
#                 out[k] = self.norm(out[k])
#             out[k] = self.padding_channel(out[k])
#         return out
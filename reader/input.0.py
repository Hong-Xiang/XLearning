"""Input for the supernet.
N.B. This routine only work on fixed size images dataset.
"""
from __future__ import absolute_import, division, print_function
import random
import os.path

import numpy as np
from six.moves import xrange

import mypylib as mp
import mypylib.image as mpli

def _patches_pair_from_tensor(self, image_high, image_low, shape, stride, n_patches=None, use_random_shuffle=True):
    """Load image(s) from a single file and generate patches
    """
    assert(len(image_high.shape)==2)
    assert(image_high.shape == image_low.shape)
    [image_heigh, image_width] = image_high.shape
    tensor = np.zeros([2, image_heigh, image_width, 1])
    tensor[0,:,:,0] = image_high
    tensor[1,:,:,0] = image_low
    patches_list_high = []
    patches_list_low = []
    for patch in mp.image.patch_generator_tensor(tensor, shape, stride, n_patches, use_random_shuffle):
        yield patch[0,:,:,:], patch[1,:,:,:]

class DataSet(object):
    def __init__(self, filename_h, filename_l,
                 patch_shape, strides,
                 is_train=True,
                 use_random_shuffle=True,
                 max_patch_image=None,
                 max_epoch=-1,
                 new_crop_method=True):
        """Constructor of DataSet object.
        Args:
        filename_h: filename of high resolution image,
        filename_l: filename of low resolution image,
        patch_shape: shape of cropped patches, must be a 2d odd integer vector
        crop_window: range of patche centers
        strides: a [Sy, Sx] vector
        """
        self._n_file = len(filename_h)
        self._filename_h = filename_h
        self._filename_l = filename_l

        self._patch_shape = patch_shape
        self._strides = strides

        self._epoch = 0
        self._max_epoch = max_epoch

        self._random = use_random_shuffle

        self._high_res_buffer = []
        self._low_res_buffer = []

        self._buffer_id = 0

        self._idfs = list(xrange(self._n_file))
        if self._random:
            random.shuffle(self._idfs)
        self._file_id = 0

        self._idps = []

        self._max_patch_image = max_patch_image

        self._is_train = is_train

        self._new_cro_method = new_crop_method

    @property
    def n_files(self):
        return self._n_file

    @property
    def files_high(self):
        return self._filename_h

    @property
    def files_low(self):
        return self._filename_l

    @property
    def height(self):
        return self._patch_shape[0]

    @property
    def width(self):
        return self._patch_shape[1]

    @property
    def epoch(self):
        """current epoch"""
        return self._epoch

    @property
    def buffer_high(self):
        return self._high_res_buffer

    @property
    def buffer_low(self):
        return self._low_res_buffer

    def read_next_file(self):
        #load new file while maximum epoch limit satisfied
        if self._max_epoch == -1 or self._epoch < self._max_epoch:

            #case last file readed: shuffle id list of file
            if self._file_id == self._n_file:
                self._epoch += 1
                self._file_id = 0
                if self._random:
                    random.shuffle(self._idfs)
            # print(self._filename_h[self._idfs[self._file_id]])
            #read images
            image_high = np.array(np.load(self._filename_h[self._idfs[self._file_id]]))
            image_low = np.array(np.load(self._filename_l[self._idfs[self._file_id]]))

            self._file_id += 1
        else:
            raise Exception("Max epoch reached!")

        if self._is_train:
            pass
            #mean_low = np.mean(image_low)
            #image_low = image_low - mean_low
            #image_high = image_high - mean_low
            #image_high = image_high[16:-16,:]
            #image_low = image_low[16:-16,:]
        return image_high, image_low

    def fill_buffer_by_cropping(self):
        image_high, image_low = self.read_next_file()
        high_res_buffer = []
        low_res_buffer = []
        #new image is loaded, crop patches
        #combine image_high and image_low into a tensor to make patches allied
        image_height, image_width = image_high.shape
        tensor = np.zeros([2, image_height, image_width, 1])
        tensor[0, :, :, 0] = image_high[:, :]
        tensor[1, :, :, 0] = image_low[:, :]

        #crop patches and fill buffer
        for patch in mpli.patch_generator_tensor(tensor,
                                                 self._patch_shape,
                                                 self._strides,
                                                 self._max_patch_image,
                                                 True):
            #patch[patch < 0] = 0
            patch_high = patch[0, :, :, 0]
            patch_low = patch[1, :, :, 0]


            # meanv = np.mean(patch_low)
            # if meanv < 1:
            #     continue
            # patch_low = patch_low -  meanv
            # patch_high = patch_high - meanv
            # stdv = np.var(patch_low)
            # patch_low = patch_low / stdv
            # patch_high = patch_high /stdv

            shape = patch_high.shape
            patch_high = np.reshape(patch_high, [1, shape[0], shape[1], 1])
            patch_low = np.reshape(patch_low, [1, shape[0], shape[1], 1])
            high_res_buffer.append(patch_high)
            low_res_buffer.append(patch_low)
        return high_res_buffer, low_res_buffer

    def fill_buffer_by_read(self):
        high_res_buffer = []
        low_res_buffer = []
        tensor_high, tensor_low = self.read_next_file()
        if tensor_high.shape[1] == self._patch_shape[0] and tensor_high.shape[2] == self._patch_shape[1]:
            n_patch = tensor_high.shape[0]
            for i in xrange(n_patch):
                high_res_buffer.append(tensor_high[i, :, :, 0])
                low_res_buffer.append(tensor_low[i, :, :, 0])
        else:
            pass
            

    def refill_buffer(self):
        self._buffer_id = 0
        if self._new_cro_method:
            high_buffer, low_buffer = self.fill_buffer_by_cropping()
        else:
            high_buffer, low_buffer = self.fill_buffer_by_read()
        self._high_res_buffer = high_buffer
        self._low_res_buffer = low_buffer

        self._idps = list(xrange(len(self._low_res_buffer)))
        random.shuffle(self._idps)

    def next_batch(self, batch_size):
        tensor_next_high = np.zeros([batch_size, self.height, self.width, 1])
        tensor_next_low = np.zeros([batch_size, self.height, self.width, 1])
        added = 0

        #while output tensor is not full filled, copy patch from buffer
        while added < batch_size:
            #if buffer is used up, fill it again
            if self._buffer_id == len(self._high_res_buffer):
               self.refill_buffer()

            #buffer is filled, output patches
            pid = self._idps[self._buffer_id]
            if len(self._low_res_buffer[0].shape) == 4:
                tensor_next_high[added, :, :, 0] = self._high_res_buffer[pid][0, :, :, 0]
                tensor_next_low[added, :, :, 0] = self._low_res_buffer[pid][0, :, :, 0]
            if len(self._low_res_buffer[0].shape) == 3:
                tensor_next_high[added, :, :, 0] = self._high_res_buffer[pid][:, :, 0]
                tensor_next_low[added, :, :, 0] = self._low_res_buffer[pid][:, :, 0]
            if len(self._low_res_buffer[0].shape) == 2:
                tensor_next_high[added, :, :, 0] = self._high_res_buffer[pid][:, :]
                tensor_next_low[added, :, :, 0] = self._low_res_buffer[pid][:, :]
            self._buffer_id += 1
            added += 1

        return tensor_next_high, tensor_next_low

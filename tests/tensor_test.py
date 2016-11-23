from __future__ import absolute_import, division, print_function
from six.moves import xrange

import numpy as np
import scipy
import h5py
import random
import pickle
import os
import unittest
import xlearn.utils.general as utg
import xlearn.utils.tensor as utt


TEST_DATA_PATH = '/home/hongxwing/Workspace/xlearn/tests/data/'


class TestLoadMat(unittest.TestCase):
    def test_basic(self):
        # TODO: Implement
        pass


class TestShapeDimFix(unittest.TestCase):

    def test_basic(self):
        inputs = (2, )
        embeds = (4, )
        output, embout = utt.shape_dim_fix(inputs, embeds)
        expect = [2]
        embexp = [4]
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))
        self.assertTrue(embout == embexp, msg=utg.errmsg(embout, embexp))

    def test_0(self):
        inputs = (2, 3)
        embeds = (1, 0, 0, 1)
        output, embout = utt.shape_dim_fix(inputs, embeds)
        expect = [1, 2, 3, 1]
        embexp = [1, 2, 3, 1]
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))
        self.assertTrue(embout == embexp, msg=utg.errmsg(embout, embexp))

    def test_mo(self):
        inputs = (2, 3)
        embeds = (-1, 0, 0, 1)
        output, embout = utt.shape_dim_fix(inputs, embeds, n_items=5)
        expect = [1, 2, 3, 1]
        embexp = [5, 2, 3, 1]
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))
        self.assertTrue(embout == embexp, msg=utg.errmsg(embout, embexp))

    def test_list(self):
        inputs = (2, 3)
        embeds = (-1, 0, 0, [2, 3])
        output, embout = utt.shape_dim_fix(inputs, embeds, n_items=5)
        expect = [1, 1, 2, 3]
        embexp = [5, 1, 2, 3]
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))
        self.assertTrue(embout == embexp, msg=utg.errmsg(embout, embexp))

    def test_list_2(self):
        inputs = (2,)
        embeds = (-1, )
        output, embout = utt.shape_dim_fix(inputs, embeds, n_items=2)
        expect = [2]
        embexp = [4]
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))
        self.assertTrue(embout == embexp, msg=utg.errmsg(embout, embexp))

    def test_list_3(self):
        inputs = (2, 3)
        embeds = (1, -1, 4, 1)
        output, embout = utt.shape_dim_fix(inputs, embeds, n_items=4)
        expect = [1, 2, 3, 1]
        embexp = [1, 8, 4, 1]
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))
        self.assertTrue(embout == embexp, msg=utg.errmsg(embout, embexp))

    def test_list_4(self):
        inputs = (2, 3)
        embeds = (1, -1, 4, 1)
        output, embout = utt.shape_dim_fix(inputs, embeds, n_items=4)
        expect = [1, 2, 3, 1]
        embexp = [1, 8, 4, 1]
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))
        self.assertTrue(embout == embexp, msg=utg.errmsg(embout, embexp))

    def test_list_5(self):
        inputs = (2, 3)
        embeds = (1, 5, -1, 1)
        output, embout = utt.shape_dim_fix(inputs, embeds, n_items=4)
        expect = [1, 2, 3, 1]
        embexp = [1, 5, 6, 1]
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))
        self.assertTrue(embout == embexp, msg=utg.errmsg(embout, embexp))

    def test_smart(self):
        inputs = (2, 3)
        embeds = (-1, 0, 0, [2, 3])
        output, embout = utt.shape_dim_fix(
            inputs, embeds, n_items=5, smart=True)
        expect = [1, 2, 3, 1]
        embexp = [5, 2, 3, 1]
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))
        self.assertTrue(embout == embexp, msg=utg.errmsg(embout, embexp))


class TestOffset1D(unittest.TestCase):

    def test_basic(self):
        """A basic use case of offset_1d"""
        length, patch_size, strides = 16, 4, 4
        output = [offset for offset in utt.offset_1d(
            length, patch_size, strides)]
        expected = [0, 4, 8, 12]
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))

    def test_basic_2(self):
        length, patch_size, strides = 4, 2, 2
        output = [offset for offset in utt.offset_1d(
            length, patch_size, strides)]
        expected = [0, 2]
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))

    def test_non_multiple(self):
        """A use case when length is not a multiple of strides"""
        length, patch_size, strides = 17, 4, 4
        output = [offset for offset in utt.offset_1d(length,
                                                     patch_size,
                                                     strides,
                                                     check_all=True)]
        expected = [0, 4, 8, 12, 13]
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))

    def test_offset0(self):
        """A use case when length is not a multiple of strides"""
        length, patch_size, strides = 16, 8, 4
        offset0 = 4
        output = [offset for offset in utt.offset_1d(length,
                                                     patch_size,
                                                     strides,
                                                     offset0=offset0)]
        expected = [4, 8]
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))

    def test_iterator(self):
        length, patch_size, strides = 8, 1, 1
        output = [offset for offset in utt.offset_1d(length,
                                                     patch_size,
                                                     strides)]
        expected = list(xrange(8))
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))

    def test_offset1(self):
        """A use case when length is not a multiple of strides"""
        length, patch_size, strides = 16, 8, 4
        offset1 = 4
        output = [offset for offset in utt.offset_1d(length,
                                                     patch_size,
                                                     strides,
                                                     offset1=offset1)]
        expected = [0, 4]
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))


class TestOffsetND(unittest.TestCase):

    def test_basic(self):
        tensor_shape = [16, 8]
        patch_shape = [8, 4]
        strides = [4, 2]
        output = [offsets for offsets in utt.offset_nd(
            tensor_shape, patch_shape, strides)]
        expected = []
        for iy in xrange(3):
            for ix in xrange(3):
                expected.append([iy * 4, ix * 2])
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))

    def test_1d(self):
        tensor_shape = [4]
        patch_shape = [2]
        strides = [2]
        output = [offset for offset in utt.offset_nd(
            tensor_shape, patch_shape, strides)]
        expect = [0, 2]
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))

    def test_split_channel(self):
        tensor_shape = [16, 16, 8]
        patch_shape = [16, 16, 1]
        strides = [1, 1, 1]
        output = [offsets for offsets in utt.offset_nd(
            tensor_shape, patch_shape, strides)]
        expected = []
        for ic in xrange(8):
            expected.append([0, 0] + [ic])
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))

    def test_split_image(self):
        tensor_shape = [5, 16, 16, 3]
        patch_shape = [1, 16, 16, 3]
        output = [offsets for offsets in utt.offset_nd(
            tensor_shape, patch_shape)]
        expected = []
        for ii in xrange(5):
            expected.append([ii] + [0, 0, 0])
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))


class TestCombineTensorList(unittest.TestCase):

    def test_1d(self):
        a = np.ones([2])
        l = [a, a]
        shape = [-1]
        output = utt.combine_tensor_list(l, shape)
        expected = np.ones([4])
        self.assertTrue(np.array_equal(output, expected),
                        msg=utg.errmsg(output, expected))

    def test_image(self):
        large_shape = (800, 800, 3)
        offset = (3, 3, 0)
        datapath = "/home/hongxwing/Workspace/xlearn/tests/data"
        input_fn = "test_combine_tensor_list.test_image.input.dat"
        output_fn = "test_combine_tensor_list.test_image.output.dat"
        with open(os.path.join(datapath, input_fn)) as file_in:
            input_ = pickle.load(file_in)
        with open(os.path.join(datapath, output_fn)) as file_ex:
            expected = pickle.load(file_ex)
        output = utt.combine_tensor_list(input_, large_shape, offset)
        self.assertTrue(np.array_equal(output, expected),
                        msg=utg.errmsg(output, expected))


class TestCropTensor(unittest.TestCase):

    def test_image(self):
        patch_shape = (64, 64, 3)
        strides = (16, 16, 1)
        input_fn = "test_crop_tensor.test_image.input.dat"
        output_fn = "test_crop_tensor.test_image.output.dat"
        with open(os.path.join(TEST_DATA_PATH, input_fn)) as file_in:
            input_ = pickle.load(file_in)
        with open(os.path.join(TEST_DATA_PATH, output_fn)) as file_ex:
            expected = pickle.load(file_ex)
        output = utt.crop_tensor(input_, patch_shape, strides)
        self.assertTrue(np.array_equal(output, expected),
                        msg=utg.errmsg(output, expected))

    def test_image_shorter(self):
        patch_shape = (64, 64, 3)
        strides = (16, 16, 1)
        input_fn = "test_crop_tensor.test_image.input.dat"
        output_fn = "test_crop_tensor.test_image.output.dat"
        with open(os.path.join(TEST_DATA_PATH, input_fn)) as file_in:
            input_ = pickle.load(file_in)
        with open(os.path.join(TEST_DATA_PATH, output_fn)) as file_ex:
            expected = pickle.load(file_ex)
        output = utt.crop_tensor(input_, patch_shape, strides, n_patches=100)
        expected = expected[:100]
        self.assertTrue(np.array_equal(output, expected),
                        msg=utg.errmsg(output, expected))


class TestDownSample1D(unittest.TestCase):

    def test_basic(self):
        inputs = np.array([1, 2, 3, 4, 5])
        output = utt.down_sample_1d(inputs, axis=0, ratio=2, method='fixed')
        expect = np.array([1, 3])
        self.assertTrue(np.array_equal(output, expect),
                        msg=utg.errmsg(output, expect))

    def test_basic_mean(self):
        inputs = np.array([1, 2, 3, 4, 5])
        output = utt.down_sample_1d(inputs, axis=0, ratio=2, method='mean')
        expect = np.array([1.5, 3.5])
        self.assertTrue(np.array_equal(output, expect),
                        msg=utg.errmsg(output, expect))


class TestDownSampleND(unittest.TestCase):

    def test_mean(self):
        input_fn = 'img000000000.npy'
        inputs = np.load(os.path.join(TEST_DATA_PATH, input_fn))
        expec_fn = 'test_down_sample_nd_test_mean.dat'
        with open(os.path.join(TEST_DATA_PATH, expec_fn), 'rb') as file_in:
            expect = pickle.load(file_in)
        output = utt.down_sample_nd(inputs, [2, 3, 1], method='mean')
        self.assertTrue(np.array_equal(output, expect),
                        msg=utg.errmsg(output, expect))

    def test_fixed(self):
        input_fn = 'img000000000.npy'
        inputs = np.load(os.path.join(TEST_DATA_PATH, input_fn))
        expec_fn = 'test_down_sample_nd_test_fixed.dat'
        with open(os.path.join(TEST_DATA_PATH, expec_fn), 'r') as file_in:
            expect = pickle.load(file_in)
        output = utt.down_sample_nd(inputs, [4, 5, 1], method='fixed')
        self.assertTrue(np.array_equal(output, expect),
                        msg=utg.errmsg(output, expect))

if __name__ == "__main__":
    unittest.main()

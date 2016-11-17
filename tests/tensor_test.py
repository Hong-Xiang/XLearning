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
import xlearn.utils.tensor as uttensor

class TestOffset1D(unittest.TestCase):
    def test_basic(self):
        """A basic use case of offset_1d"""
        length, patch_size, strides = 16, 4, 4
        output = [offset for offset in uttensor.offset_1d(length, patch_size, strides)]
        expected = [0, 4, 8, 12]
        self.assertTrue(output == expected,
                        msg="Output {0} while expected {1}.".format(output, expected))

    def test_non_multiple(self):
        """A use case when length is not a multiple of strides"""
        length, patch_size, strides = 17, 4, 4
        output = [offset for offset in uttensor.offset_1d(length,
                                                          patch_size,
                                                          strides,
                                                          check_all=True)]
        expected =  [0, 4, 8, 12, 13]
        self.assertTrue(output == expected,
                        msg="Output {0} while expected {1}.".format(output, expected))

    def test_offset0(self):
        """A use case when length is not a multiple of strides"""
        length, patch_size, strides = 16, 8, 4
        offset0 = 4
        output = [offset for offset in uttensor.offset_1d(length,
                                                          patch_size,
                                                          strides,
                                                          offset0=offset0)]
        expected =  [4, 8]
        self.assertTrue(output == expected,
                        msg="Output {0} while expected {1}.".format(output, expected))

    def test_iterator(self):
        length, patch_size, strides = 8, 1, 1
        output = [offset for offset in uttensor.offset_1d(length,
                                                          patch_size,
                                                          strides)]
        expected = list(xrange(8))
        self.assertTrue(output == expected,
                        msg="Output {0} while expected {1}.".format(output, expected))

    def test_offset1(self):
        """A use case when length is not a multiple of strides"""
        length, patch_size, strides = 16, 8, 4
        offset1 = 4
        output = [offset for offset in uttensor.offset_1d(length,
                                                          patch_size,
                                                          strides,
                                                          offset1=offset1)]
        expected =  [0, 4]
        self.assertTrue(output == expected,
                        msg="Output {0} while expected {1}.".format(output, expected))

class TestOffsetND(unittest.TestCase):
    def test_basic(self):
        tensor_shape = [16, 8]
        patch_shape = [8, 4]
        strides = [4, 2]
        output = [offsets for offsets in uttensor.offset_nd(tensor_shape, patch_shape, strides)]
        expected = []
        for iy in xrange(3):
            for ix in xrange(3):
                expected.append([iy*4, ix*2])
        self.assertTrue(output == expected,
                        msg="Output {0} while expected {1}.".format(output, expected))

    def test_split_channel(self):
        tensor_shape = [16, 16, 8]
        patch_shape = [16, 16, 1]
        strides = [1, 1, 1]
        output = [offsets for offsets in uttensor.offset_nd(tensor_shape, patch_shape, strides)]
        expected = []
        for ic in xrange(8):
            expected.append([0, 0]+[ic])
        self.assertTrue(output == expected,
                        msg="Output {0} while expected {1}.".format(output, expected))

    def test_split_image(self):
        tensor_shape = [5, 16, 16, 3]
        patch_shape = [1, 16, 16, 3]
        output = [offsets for offsets in uttensor.offset_nd(tensor_shape, patch_shape)]
        expected = []
        for ii in xrange(5):
            expected.append([ii] + [0, 0, 0])
        self.assertTrue(output == expected,
                        msg="Output {0} while expected {1}.".format(output, expected))

class TestCombineTensorList(unittest.TestCase):
    def test_1d(self):
        a = np.ones([2])
        l = [a, a]
        shape = [4]
        output = uttensor.combine_tensor_list(l, shape)
        expected = np.ones([4])
        self.assertTrue(np.array_equal(output,expected),
                        msg="Output {0} while expected {1}.".format(output, expected))

    def test_image(self):
        large_shape = (800, 800, 3)
        offset = (3, 3, 0)
        datapath = "/home/hongxwing/Workspace/xlearn/tests/data"
        input_fn = "test_combine_tensor_list.test_image.input.dat"
        output_fn = "test_combine_tensor_list.test_image.output.dat"
        with open(os.path.join(datapath, input_fn)) as fi:
            input_ = pickle.load(fi)
        with open(os.path.join(datapath, output_fn)) as fi:
            expected = pickle.load(fi)
        output = uttensor.combine_tensor_list(input_, large_shape, offset)
        self.assertTrue(np.array_equal(output, expected),
                        msg="Output {0} while expected {1}.".format(output, expected))

class TestCropTensor(unittest.TestCase):
    def test_image(self):
        patch_shape = (64, 64, 3)
        strides = (16, 16, 1)
        datapath = "/home/hongxwing/Workspace/xlearn/tests/data"
        input_fn = "test_crop_tensor.test_image.input.dat"
        output_fn = "test_crop_tensor.test_image.output.dat"
        with open(os.path.join(datapath, input_fn)) as fi:
            input_ = pickle.load(fi)
        with open(os.path.join(datapath, output_fn)) as fi:
            expected = pickle.load(fi)
        output = uttensor.crop_tensor(input_, patch_shape, strides)
        self.assertTrue(np.array_equal(output, expected),
                        msg="Output {0} while expected {1}.".format(output, expected))

if __name__=="__main__":
    unittest.main()
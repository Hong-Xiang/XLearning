from __future__ import absolute_import, division, print_function

import numpy as np
import scipy
import h5py
import random
import pickle
import os
import unittest
import xlearn.utils.general as utg
import xlearn.utils.image as uti

TEST_DATA_PATH = '/home/hongxwing/Workspace/xlearn/tests/data/'

class TestImage2Tensor(unittest.TestCase):
    def test_basic(self):
        img = np.load(os.path.join(TEST_DATA_PATH, 'img000000001.npy'))
        img = np.array(img)
        tensor = uti.image2tensor(img)
        output = tensor.shape
        expect = tuple([1, img.shape[0], img.shape[1], img.shape[2]])
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))

class TestRGB2Gray(unittest.TestCase):
    def test_basic(self):
        img = np.load(os.path.join(TEST_DATA_PATH, 'img000000001.npy'))
        img = np.array(img)
        tensor = uti.image2tensor(img)
        output = uti.rgb2gray(tensor)
        with open(os.path.join(TEST_DATA_PATH, 'test_image2tensor_basic.dat')) as fileout:
            expect = pickle.load(fileout)
        self.assertTrue(np.array_equal(output, expect), msg=utg.errmsg(output, expect))


if __name__=="__main__":
    unittest.main()


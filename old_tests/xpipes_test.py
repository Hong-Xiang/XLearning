from __future__ import absolute_import, division, print_function
from six.moves import xrange

import numpy as np
import scipy
import h5py
import random
import pickle
import os
import unittest
import copy
import xlearn.utils.general as utg
import xlearn.utils.xpipes as utp

TEST_DATA_PATH = '/home/hongxwing/Workspace/xlearn/tests/data/'


class TestPipe(unittest.TestCase):

    def test_merge(self):
        pipe_const0 = utp.Inputer(item='test')
        pipe_const1 = utp.Inputer(item='test')
        pipe_const2 = utp.Inputer(item='test')
        pipe_merge = utp.Pipe([pipe_const0, pipe_const1, pipe_const2])
        output = next(pipe_merge.out)
        expected = ['test', 'test', 'test']
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))

    def test_is_start(self):
        pipe_start = utp.Pipe(is_start=True)
        output = []
        for i in range(3):
            output.append(next(pipe_start.out))
        expected = [None, None, None]
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))

    def test_stack_tensor(self):
        input1 = np.ones([3, 3])
        input2 = np.ones([4, 4]) * 2
        pipe_input1 = utp.Inputer(input1)
        pipe_input2 = utp.Inputer(input2)
        pipe_stack = utp.Pipe([pipe_input1, pipe_input2])
        output = next(pipe_stack.out)
        expect = [input1, input2]
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))


class TestCounter(unittest.TestCase):

    def test_basic(self):
        pipe_count = utp.Counter()
        output = []
        for i in xrange(3):
            output.append(next(pipe_count.out))
        expected = [0, 1, 2]
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))

    def test_max(self):
        pipe_count = utp.Counter()
        pipe_count.max_state = 5
        output = [cnt for cnt in pipe_count.out]
        expected = [0, 1, 2, 3, 4]
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))


class TestInput(unittest.TestCase):

    def test_basic(self):
        pipe_input = utp.Inputer(item='test', const_output=True)
        output = []
        for i in xrange(3):
            output.append(next(pipe_input.out))
        expected = ['test', 'test', 'test']
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))

    def test_item(self):
        pipe_input = utp.Inputer()
        pipe_input.insert('test1')
        pipe_input.insert('test2')
        pipe_input.insert('test3')
        output = [item for item in pipe_input.out]
        expected = ['test1', 'test2', 'test3']
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))


class TestCopyer(unittest.TestCase):

    def test_basic(self):
        pipe_input = utp.Inputer(item='test')
        pipe_copy = utp.Copyer(pipe_input, copy_number=3)
        output = [item for item in pipe_copy.out]
        expected = ['test', 'test', 'test']
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))

    def test_set_copy_number(self):
        pipe_input = utp.Inputer(item='test')
        pipe_copy = utp.Copyer(pipe_input)
        pipe_copy.copy_number = 3
        output = [item for item in pipe_copy.out]
        expected = ['test', 'test', 'test']
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))


class TestNPYReader(unittest.TestCase):
    pass


class TestTensorFormater(unittest.TestCase):

    def test_auto_imgrgb(self):
        data_file = os.path.join(TEST_DATA_PATH, 'img000000001.npy')
        imgrgb = np.array(np.load(data_file))
        shapergb = imgrgb.shape
        pipe_input = utp.Inputer(imgrgb)
        pipe_formater = utp.TensorFormater(pipe_input)
        imgtensor = next(pipe_formater.out)
        output = imgtensor.shape
        expect = [1] + list(shapergb)
        expect = tuple(expect)
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))

    def test_auto_stacked_image(self):
        data_file = os.path.join(TEST_DATA_PATH, 'img000000001.npy')
        imgrgb = np.array(np.load(data_file))
        input_ = np.zeros(
            [2, imgrgb.shape[0], imgrgb.shape[1], imgrgb.shape[2]])
        input_[0, :, :, :] = imgrgb
        input_[1, :, :, :] = imgrgb

        pipe_input = utp.Inputer(input_)
        pipe_tensor = utp.TensorFormater(pipe_input)
        tensor = next(pipe_tensor.out)
        output = tensor.shape
        expect = (2, 288, 352, 3)
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))

    def test_auto_after_stacker(self):
        data_file = os.path.join(TEST_DATA_PATH, 'img000000001.npy')
        imgrgb = np.array(np.load(data_file))
        imgin = np.reshape(imgrgb, [1, 288, 352, 3])

        pipe_a = utp.Inputer(imgin)
        pipe_b = utp.Inputer(imgin)

        pipe_merge = utp.Pipe([pipe_a, pipe_b])
        pipe_stacker = utp.TensorStacker(pipe_merge)
        pipe_tensor = utp.TensorFormater(pipe_stacker)
        tensor = next(pipe_tensor.out)
        output = tensor.shape
        expect = (2, 288, 352, 3)
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))

    def test_auto_after_stacker_gray(self):
        data_file = os.path.join(TEST_DATA_PATH, 'img000000001.npy')
        imgrgb = np.array(np.load(data_file))
        imggray = imgrgb[:, :, 0]
        imgin = np.reshape(imggray, [1, 288, 352, 1])

        pipe_a = utp.Inputer(imgin)
        pipe_b = utp.Inputer(imgin)
        pipe_merge = utp.Pipe([pipe_a, pipe_b])
        pipe_stacker = utp.TensorStacker(pipe_merge)
        pipe_tensor = utp.TensorFormater(pipe_stacker)
        tensor = next(pipe_tensor.out)
        output = tensor.shape
        expect = (2, 288, 352, 1)
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))


class TestTensorStacker(unittest.TestCase):

    def test_basic(self):
        pipe_a = utp.Inputer(np.ones([3, 3]))
        pipe_b = utp.Inputer(np.zeros([3, 3]))
        pipe_merge = utp.Pipe([pipe_a, pipe_b])
        pipe_stacker = utp.TensorStacker(pipe_merge)
        stacked = next(pipe_stacker.out)
        output = stacked.shape
        expect = (2, 3, 3)
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))

    def test_one(self):
        pipe_a = utp.Inputer(np.ones([1, 10, 10, 3]))
        pipe_b = utp.Inputer(np.zeros([1, 10, 10, 3]))
        pipe_merge = utp.Pipe([pipe_a, pipe_b])
        pipe_stacker = utp.TensorStacker(pipe_merge)
        stacked = next(pipe_stacker.out)
        output = stacked.shape
        expect = (2, 10, 10, 3)
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))

    def test_gray(self):
        pipe_a = utp.Inputer(np.ones([1, 10, 10, 1]))
        pipe_b = utp.Inputer(np.zeros([1, 10, 10, 1]))
        pipe_merge = utp.Pipe([pipe_a, pipe_b])
        pipe_stacker = utp.TensorStacker(pipe_merge)
        stacked = next(pipe_stacker.out)
        output = stacked.shape
        expect = (2, 10, 10, 1)
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))
    
    def test_gray2(self):
        data_file = os.path.join(TEST_DATA_PATH, 'img000000001.npy')
        imgrgb = np.array(np.load(data_file))
        imggray = imgrgb[:, :, 0]
        imgin = np.reshape(imggray, [1, 288, 352, 1])

        pipe_a = utp.Inputer(imgin)
        pipe_b = utp.Inputer(imgin)
        pipe_merge = utp.Pipe([pipe_a, pipe_b])
        pipe_stacker = utp.TensorStacker(pipe_merge)
        stacked = next(pipe_stacker.out)
        output = stacked.shape
        expect = (2, 288, 352, 1)
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))

if __name__ == "__main__":
    unittest.main()

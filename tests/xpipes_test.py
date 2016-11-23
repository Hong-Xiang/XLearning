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
import xlearn.utils.xpipes as xpipes


class TestPipe(unittest.TestCase):

    def test_merge(self):
        pipe_const0 = xpipes.Inputer(item='test')
        pipe_const1 = xpipes.Inputer(item='test')
        pipe_const2 = xpipes.Inputer(item='test')
        pipe_merge = xpipes.Pipe([pipe_const0, pipe_const1, pipe_const2])
        output = next(pipe_merge.out)
        expected = ['test', 'test', 'test']
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))

    def test_is_start(self):
        pipe_start = xpipes.Pipe(is_start=True)
        output = []
        for i in range(3):
            output.append(next(pipe_start.out))
        expected = [None, None, None]
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))


class TestCounter(unittest.TestCase):

    def test_basic(self):
        pipe_count = xpipes.Counter()
        output = []
        for i in xrange(3):
            output.append(next(pipe_count.out))
        expected = [0, 1, 2]
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))

    def test_max(self):
        pipe_count = xpipes.Counter()
        pipe_count.max_state = 5
        output = [cnt for cnt in pipe_count.out]
        expected = [0, 1, 2, 3, 4]
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))


class TestInput(unittest.TestCase):

    def test_basic(self):
        pipe_input = xpipes.Inputer(item='test', const_output=True)
        output = []
        for i in xrange(3):
            output.append(next(pipe_input.out))
        expected = ['test', 'test', 'test']
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))

    def test_item(self):
        pipe_input = xpipes.Inputer()
        pipe_input.insert('test1')
        pipe_input.insert('test2')
        pipe_input.insert('test3')
        output = [item for item in pipe_input.out]
        expected = ['test1', 'test2', 'test3']
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))


class TestCopyer(unittest.TestCase):

    def test_basic(self):
        pipe_input = xpipes.Inputer(item='test')
        pipe_copy = xpipes.Copyer(pipe_input, copy_number=3)
        output = [item for item in pipe_copy.out]
        expected = ['test', 'test', 'test']
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))

    def test_set_copy_number(self):
        pipe_input = xpipes.Inputer(item='test')
        pipe_copy = xpipes.Copyer(pipe_input)
        pipe_copy.copy_number = 3
        output = [item for item in pipe_copy.out]
        expected = ['test', 'test', 'test']
        self.assertTrue(output == expected, msg=utg.errmsg(output, expected))


class TestNPYReader(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()

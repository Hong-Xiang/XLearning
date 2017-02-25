from __future__ import absolute_import, division, print_function
import unittest

import xlearn.utils.general as utg


class TestSeperateFileName(unittest.TestCase):

    def test_basic_0(self):
        output = utg.seperate_file_name("sino000000018.raw")
        expect = ('sino', 18, 'raw')
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))

    def test_basic_1(self):
        output = utg.seperate_file_name("sino000000018")
        expect = ('sino', 18, '')
        self.assertTrue(output == expect, msg=utg.errmsg(output, expect))

if __name__ == "__main__":
    unittest.main()

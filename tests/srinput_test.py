from __future__ import absolute_import, division, print_function
import unittest

import xlearn.reader.srinput as sri

class TestSRInput(unittest.TestCase):
    def test_conf_file(self):
        sri.config_file_generator("conf.json",
                                  path_data="~/Workspace/Datas/nature_image",
                                  prefix_data="img",
                                  path_label="~/Workspace/Datas/nature_image",
                                  prefix_label="img",
                                  patch_shape=[1, 33, 33, 1],
                                  strides=[1, 10, 10, 1],
                                  batch_size=128)
    
    def test_data_set(self):
        pass


if __name__ == "__main__":
    unittest.main()
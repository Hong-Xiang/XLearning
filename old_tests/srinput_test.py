from __future__ import absolute_import, division, print_function
import unittest
import os
import pickle
import numpy as np
import xlearn.utils.general as utg
import xlearn.reader.srinput as sri


TEST_DATA_PATH = '/home/hongxwing/Workspace/xlearn/tests/data/'


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
        fullname = os.path.join(TEST_DATA_PATH, "conf.json")
        dataset = sri.DataSetSR(conf=fullname)
        img_l, img_h = dataset.next_batch()
        exp_l_fn = os.path.join(TEST_DATA_PATH, "test_srinput_dataset.low.dat")
        exp_h_fn = os.path.join(
            TEST_DATA_PATH, "test_srinput_dataset.high.dat")
        with open(exp_l_fn, 'r') as fnlow:
            expect_l = pickle.load(fnlow)
        with open(exp_h_fn, 'r') as fnhigh:
            expect_h = pickle.load(fnhigh)
        self.assertTrue(np.array_equal(expect_l, img_l),
                        utg.errmsg(img_l, expect_l))
        self.assertTrue(np.array_equal(expect_h, img_h),
                        utg.errmsg(img_h, expect_h))

if __name__ == "__main__":
    unittest.main()

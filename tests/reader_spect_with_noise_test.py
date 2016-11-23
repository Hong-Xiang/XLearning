from __future__ import absolute_import, division, print_function
import unittest
import xlearn.reader.spect_with_noise as rsn
import xlearn.utils.general as utg
import xlearn.utils.xpipes as utp

class TestFileNameTasks(unittest.TestCase):
    def test_former(self):
        proj_id = 34
        depth = 11
        noise_level = 4
        case_id = 22
        output = rsn.filename_former(proj_id, depth, noise_level, case_id)
        expect = "sino034110422.npy"
        self.assertTrue(output==expect, msg=utg.errmsg(output, expect))

    def test_parser(self):
        proj_id = 3
        depth = 15
        noise_level = 16
        case_id = 27
        output = rsn.filename_parser("sino003151627.npy")
        expect = (proj_id, depth, noise_level, case_id)
        self.assertTrue(output==expect, msg=utg.errmsg(output, expect))

class TestReaderFileNameSystem(unittest.TestCase):
    def test_basic(self):
        data_path = "/home/hongxwing/Workspace/Datas/SPECT_noise_train/"
        pipe_filename = utp.FileNameLooper(data_path, prefix='sino')
        pipe_pair = rsn.DataLableFilePair(pipe_filename)
        pipe_buffer = utp.Buffer(pipe_pair)
        pipe_data = utp.Pipe(pipe_buffer)
        pipe_label = utp.Pipe(pipe_buffer)
        output1 = next(pipe_data.out)
        output2 = next(pipe_label.out)
        expect1 = ['/home/hongxwing/Workspace/Datas/SPECT_noise_train/sino000000400.npy']
        expect2 = ['/home/hongxwing/Workspace/Datas/SPECT_noise_train/sino000000000.npy']

if __name__=="__main__":
    unittest.main()

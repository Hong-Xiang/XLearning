import numpy as np
from xlearn.datasets.rawfile import NPYZ, HDF5

def test_npyz_npy():
    rawfile = NPYZ('/home/hongxwing/Workspace/Datas/tests/arange1024.npy')
    rawfile.init()
    std_data = np.arange(1024)
    std_data = np.reshape(std_data, [-1, 1])
    assert np.array_equal(rawfile.datasets['data'], std_data)
    rawfile.final()

def test_npyz_npz():
    rawfile = NPYZ('/home/hongxwing/Workspace/Datas/tests/arange1024.npz', ['data'])
    rawfile.init()
    std_data = np.arange(1024)
    std_data = np.reshape(std_data, [-1, 1])
    assert np.array_equal(rawfile.datasets['data'], std_data)
    rawfile.final()

def test_h5py():
    rawfile = HDF5('/home/hongxwing/Workspace/Datas/tests/arange1024.h5', ['data'])
    rawfile.init()
    std_data = np.arange(1024)
    std_data = np.reshape(std_data, [-1, 1])
    assert np.array_equal(rawfile.datasets['data'], std_data)
    rawfile.final()

# if __name__ == "__main__":
#     test_npyz_npy()
#     test_npyz_npz()
#     test_h5py()
import numpy as np
import random
import xlearn.datasets.loaders as lds
import xlearn.datasets.rawfile as rf
import nose

def test_sampler_range():
    id_min = 0
    id_max = 10
    idx_sampler = lds.IndexSampler(is_shuffle=False, idx_file=None, idx_range=[id_min, id_max])
    outs = [next(idx_sampler) for _ in range(10)]
    assert outs == list(range(10))

def test_sampler_idxs():
    idx_file = '/home/hongxwing/Workspace/Datas/tests/arange1024.npy'
    idx_sampler = lds.IndexSampler(is_shuffle=False, idx_file=idx_file)
    outs = [next(idx_sampler) for _ in range(10)]
    assert outs == list(range(10))

def test_load():
    data = np.random.uniform(size=[10, 3, 3])
    dataset = {'data': data}
    idx_sampler = lds.IndexSampler(is_shuffle=False, idx_file=None, idx_range=[0, 10])
    ldr = lds.Loader(dataset, idx_sampler)    
    for i in range(10):    
        assert np.array_equal(next(ldr)['data'], data[i, ...])

def test_load_raw_file():
    from pdb import set_trace
    raw_file =  rf.NPYZ('/home/hongxwing/Workspace/Datas/tests/arange1024.npz', keys=['data'])
    print(raw_file.datasets)
    idx_sampler = lds.IndexSampler(is_shuffle=False, idx_file=None, idx_range=[0, 10])
    
    ldr = lds.Loader(raw_file.datasets, idx_sampler)
    raw_file.init()
    for i in range(10):    
        assert np.array_equal(next(ldr)['data'], np.array([i]))
    raw_file.final()

if __name__ == "__main__":
    from xlearn.utils.general import enter_debug
    enter_debug()
    test_load_raw_file()
    pass

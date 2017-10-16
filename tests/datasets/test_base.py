import tempfile
import json
import nose
import numpy as np
import nose
import tensorflow as tf
import xlearn.datasets.base_new as db
import os

def test_raw_auto():
    raw_auto = db.RawdataAuto('phantom_sinograms_3.h5', ['sinograms'], 'train')
    raw_auto.init()
    for i in range(10):
        s = next(raw_auto)    
        assert s['sinograms'].shape == (320, 640, 1)
        assert type(s['idx']) == np.int64
    raw_auto.final()


def test_dataset():    
    rfn = 'phantom_sinograms_3.h5'
    rfk = ['sinograms']
    keys_out = ['data', 'idx']
    keys_map = {'data': 'sinograms', 'idx': 'idx'}
    keys_pass = ['data', 'idx']
    proc_args = {'keys_out': keys_out, 'keys_map': keys_map, 'keys_pass': keys_pass}
    ds = db.Dataset(mode='train', raw_file_name=rfn, raw_file_keys=rfk, proc_names=['Proc'], proc_kwargs=[proc_args], batch_size=8)
    ds.init()
    for i in range(100):
        s = next(ds)
        assert list(s.keys()) == ['data', 'idx']
        assert s['data'].shape == (8, 320, 640, 1)
        assert s['idx'].shape == (8, )
    ds.final()


def test_dataset_with():    
    rfn = 'phantom_sinograms_3.h5'
    rfk = ['sinograms']
    keys_out = ['data', 'idx']
    keys_map = {'data': 'sinograms', 'idx': 'idx'}
    keys_pass = ['data', 'idx']
    proc_args = {'keys_out': keys_out, 'keys_map': keys_map, 'keys_pass': keys_pass}
    with db.Dataset(mode='train', raw_file_name=rfn, raw_file_keys=rfk, proc_names=['Proc'], proc_kwargs=[proc_args], batch_size=8) as ds:    
        for i in range(10):
            s = next(ds)
            assert list(s.keys()) == ['data', 'idx']
            assert s['data'].shape == (8, 320, 640, 1)
            assert s['idx'].shape == (8, )

def test_tf():    
    rfn = 'phantom_sinograms_3.h5'
    rfk = ['sinograms']
    keys_out = ['data', 'idx']
    keys_map = {'data': 'sinograms', 'idx': 'idx'}
    keys_pass = ['data', 'idx']
    proc_args = {'keys_out': keys_out, 'keys_map': keys_map, 'keys_pass': keys_pass}
    with db.Dataset(mode='train', raw_file_name=rfn, raw_file_keys=rfk, proc_names=['Proc'], proc_kwargs=[proc_args], batch_size=8, is_tf=True) as ds:    
        data_op, idx_op = ds.nodes['data'], ds.nodes['idx']
        os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
        sv = tf.train.Supervisor(logdir="logdir")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with sv.managed_session(config=config) as sess:
            for i in range(10):
                s = sess.run({'data': data_op, 'idx': idx_op})            
                assert list(s.keys()) == ['data', 'idx']
                assert s['data'].shape == (8, 320, 640, 1)
                assert s['idx'].shape == (8, )
    
# def test_sample():
#     with tempfile.NamedTemporaryFile(mode='w+') as json_file:
#         rfn = 'arange1024.npy'
#         jfn = json_file.name                
#         configs = {
#             'raw_file_name': rfn,
#             'raw_type': 'npyz',
#             'raw_keys': None,
#             'batch_size': 32
#         }
#         json.dump(configs, json_file)
#         json_file.seek(0)
#         with base.Dataset(config_files=jfn) as dataset:
#             ss = dataset.sample()
#             assert ss.shape == [32, 1]

if __name__ == "__main__":
    import cProfile
    import pstats
    # cProfile.run('test_dataset()', 'stats')
    p = pstats.Stats('stats')
    p.sort_stats('tottime')
    p.print_stats()
    pass
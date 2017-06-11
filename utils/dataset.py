import h5py

def analysis_dataset(h5file_name):
    with h5py.File(h5file_name, 'r') as fin:
        dataset_keys = list(fin.keys())
        out = {}
        for k in dataset_keys:
            out[k] = dict()
            out[k]['shape'] = fin[k].shape            
            out[k]['attrs'] = dict(fin[k].attrs)
    return out


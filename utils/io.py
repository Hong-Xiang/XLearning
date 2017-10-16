import h5py
import scipy.io
import struct
import numpy as np
from pathlib import Path
import json

def load_jsons(file_names):
    cfgs = dict()
    for fn in reversed(file_names):
        p = Path(fn)
        with open(p, 'r') as fin:
            cfgs.update(json.load(fin))
        
def loadmat(file_name, var_names=None):
    """ Load Matlab .mat files into a dict.
    old versions (<v7.3 and v7.3 is supported)
    """
    #TODO: Impl
    try:
        pass
    except:
        pass

def loadbin(file_name, shape, fmt=None):
    if fmt is None:
        fmt = "<f"
    with open(file_name, 'rb') as fin:
        file_content = fin.read()
        data = list(struct.iter_unpack(fmt, file_content))
        data = np.array(data)
        data = data.reshape(shape)
    return data


def load_bin(file_name, shape, fmt=None, order='c'):
    """ Load nd-array bindary data.
    """
    if fmt is None:
        fmt = "<f"
    with open(file_name, 'rb') as fin:
        file_content = fin.read()
        data = list(struct.iter_unpack(fmt, file_content))
        data = np.array(data)
        data = data.reshape(shape)
    return data


def save_bin(file_name, data, fmt=None, order='c'):
    """ Save nd-array bindary data.
    """
    if fmt is None:
        fmt = "<f"
    with open(file_name, 'wb') as fout:
        to_pack = np.reshape(data, [-1])
        nb_data = to_pack.shape[0]
        for num in to_pack:
            bin_num = struct.pack(fmt, num)
            fout.write(bin_num)
    return data
import h5py
import numpy as np
import pathlib
from xlearn.utils.options import auto_configs

def get_instance(type_name, **kwargs):
    name2cls = {
        'hdf5': HDF5,
        'HDF5': HDF5,
        'npyz': NPYZ,
        'NPYZ': NPYZ
    }
    if not type_name in name2cls:
        raise ValueError("Raw file reader {name} not found.".format(name=type_name))
    return name2cls[type_name](**kwargs)

class RawFile:    
    @auto_configs()
    def __init__(self, file_path, keys, name='RawFile'):
        self.file_path = pathlib.Path(file_path)
        if not self.file_path.is_file():
            raise ValueError("Invalid file path {}.".format(file_path))
        self.keys = keys
        self.datasets = dict()    
        self.name = name
        
        
    
    def init(self):
        self._init_impl()

    def final(self):
        self._final_impl()

    def _init_impl(self):
        raise NotImplementedError

    def _final_impl(self):
        pass

    def __enter__(self):
        init()
        return self
    
    def __exit__(self, etype, value, traceback):
        final()
    


class HDF5(RawFile):        
    @auto_configs()
    def __init__(self, file_path, keys, is_full_load=False, name='RawFile/HDF5'):
        super(HDF5, self).__init__(file_path, keys, name=name)        

    def _init_impl(self):
        self.__raw_file = h5py.File(self.file_path, 'r')
        for k in self.keys:
            if self.c.is_full_load:
                self.datasets[k] = np.array(self.__raw_file[k])
            else:
                self.datasets[k] = self.__raw_file[k]

    def _final_impl(self):
        self.__raw_file.close()
        self.__raw_file = None        

class NPYZ(RawFile):    
    @auto_configs()
    def __init__(self, file_path, keys=None, name='RawFile/NPYZ'):
        self.is_npz = True
        if str(file_path).endswith('.npy'):
            self.is_npz = False
            if keys is None:
                keys = ['data']
        super(NPYZ, self).__init__(file_path, keys, name=name)

    def _init_impl(self):        
        self.__raw_data = np.load(str(self.file_path))  
        if self.is_npz:      
            for k in self.keys:
                self.datasets[k] = self.__raw_data[k]
        else:
            self.datasets[self.keys[0]] = self.__raw_data
        

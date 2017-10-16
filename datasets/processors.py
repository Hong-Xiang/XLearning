import numpy as np
from xlearn.utils.options import json_config, auto_configs

def get_instance(name, *args, **kwargs):
    name2cls = {
        'proc': Proc,
        'Proc': Proc,
        'norm': Norm,
        'crop': Crop
    }
    return name2cls[name](*args, **kwargs)

class Proc:
    def __init__(self, keys_out, keys_map=None, pass_keys=None):
        self.keys_out = keys_out
        if keys_map is None:
            self.keys_map = {k: k for k in keys_out}
        else:
            self.keys_map = keys_map
        if pass_keys is None:
            pass_keys = tuple()
        self.pass_keys = pass_keys

    def _proc(self, key_out, key_in, data):
        return data

    def __call__(self, data_dict):
        out = dict()
        for k in self.keys_out:
            if k in self.pass_keys:
                out[k] = data_dict[self.keys_map[k]]
            else:
                out[k] = self._proc(k, self.keys_map[k], data_dict[self.keys_map[k]])
        return out

class Norm(Proc):
    def __init__(self, keys_out, keys_map=None, pass_keys=None,
                 gamma=1.0,
                 mean=0.0,
                 std=1.0):
        super(Norm, self).__init__(keys_out, keys_map, pass_keys)
        self.gamma = gamma
        self.mean = mean
        self.std = std

    def norm(self, tensor, mean_value=None, std_value=None):
        """ Normalization datas
        """
        normed = numpy.array(tensor)        
        if mean_value is None:
            mean_value = self.mean
        if std_value is None:
            std_value = self.std

        normed = (normed - mean_value) / std_value
        if not self.gamma == 1.0:
            normed = np.power(normed, self.gamma)
        return normed

    def denorm(self, tensor, mean_value=None, std_value=None):
        denormed = numpy.array(tensor)        
        if mean_value is None:
            mean_value = self.mean
        if std_value is None:
            std_value = self.std
        if not self.gamma == 1.0:
            denormed = numpy.power(denormed, 1.0 / self.gamma)
        denormed = denormed * std_value + mean_value
        return denormed

    def __call__(self, data_dict, mean_value=None, std_value=None):
        return {k: self.norm(data_dict[k], mean_value, std_value) if k in self.keys else data_dict[k] for k in data_dict}

class PoissonNoise(Proc):    
    def __init__(self, keys_out, keys_map=None, pass_keys=None,
                 nb_events):
        super(PoissonNoise, self).__init__(keys_out, keys_map, pass_keys)
        self.nb_events = nb_events
    
    def __call__(self, data_dict):
        out = dict()
        for k in self.keys_out:
            if k in self.pass_keys:
                out[k] = data_dict[self.keys_map[k]]
            else:
                out[k] = self._proc(k, self.keys_map[k], data_dict[self.keys_map[k]])
        return out

class Crop(Proc):
    pass
        


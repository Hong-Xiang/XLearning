import numpy as np
from .base import DataSetImages
from ..utils.general import with_config

class MNISTImage(DataSetImages):
    @with_config
    def __init__(self, dataset_name='mnist', **kwargs):
        DataSetImages.__init__(self, dataset_name='mnist', **kwargs)

class Celeba(DataSetImages):
    @with_config
    def __init__(self, dataset_name='celeba', **kwargs):
        DataSetImages.__init__(self, dataset_name='celeba', **kwargs)

class Flickr25k(DataSetImages):
    @with_config
    def __init__(self, dataset_name='flickr25k', **kwargs):
        DataSetImages.__init__(self, dataset_name='flickr25k', **kwargs)

class SinoShep(DataSetImages):
    @with_config
    def __init__(self, dataset_name='sino_shep', **kwargs):
        DataSetImages.__init__(self, dataset_name='sino_shep', **kwargs)        
        nb_padding = int(np.ceil(self.p.crop_shape[1]/360))
        self.params['padding'] = [1, nb_padding]
        self.params.update_short_cut()

class SinoShepTest(DataSetImages):
    @with_config
    def __init__(self, dataset_name='sino_shep_test', **kwargs):
        DataSetImages.__init__(self, dataset_name='sino_shep_test', **kwargs)        
        nb_padding = int(np.ceil(self.p.crop_shape[1]/360))
        self.params['padding'] = [1, nb_padding]
        self.params.update_short_cut()

class SinoDero(DataSetImages):
    @with_config
    def __init__(self, dataset_name='sino_dero', **kwargs):
        DataSetImages.__init__(self, dataset_name='sino_dero', **kwargs)        
        nb_padding = int(np.ceil(self.p.crop_shape[1]/360))
        self.params['padding'] = [1, nb_padding]
        self.params.update_short_cut()

class PETRebin(DataSetImages):
    @with_config
    def __init__(self, dataset_name='sino_rebin', **kwargs):
        DataSetImages.__init__(self, dataset_name='sino_rebin', **kwargs)        
        nb_padding = int(np.ceil(self.p.crop_shape[1]/320))
        self.params['padding'] = [1, nb_padding]
        self.params.update_short_cut()

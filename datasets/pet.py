import numpy as np
from .base import DataSetImages
from ..utils.general import with_config

class PETRebin(DataSetImages):
    @with_config
    def __init__(self, dataset_name='sino_rebin', **kwargs):
        DataSetImages.__init__(self, dataset_name='sino_rebin', **kwargs)        
        nb_padding = int(np.ceil(self.p.crop_shape[1]/320))
        self.params['padding'] = [1, nb_padding]
        self.params.update_short_cut()

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
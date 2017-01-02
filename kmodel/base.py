"""Base class for keras based models.
"""

import xlearn.utils.general as utg

class KNet(object):
    """Base class for keras nets.
    """
    def __init__(self, filenames=None, **kwargs):
        self._settings = utg.merge_settings(settings=None, filenames=filenames, **kwargs)
        self._gather_settings()
        self._model = self._net_definition()

    def _gather_settings(self):
        self._batch_size = self._settings['batch_size']
        self._input_dim = self._settings['input_dim']

    def _net_definition(self):
        pass

    @property
    def model(self):
        return self._model

    
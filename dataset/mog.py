""" Mixture of Gaussian data """

from .base import DataSetBase


class MoG(DataSetBase):

    def __init__(self, sigma=1, data_dim=2, nb_center=8, mus=None, **kwargs):
        super(MoG, self).__init__(sigma=sigma,
                                  data_dim=data_dim, nb_center=nb_center, mus=mus, **kwargs)
        self._data_dim = self._settings['data_dim']
        self._nb_center = self._settings['nb_center']
        self._mus = self._settings['mus']
        self._sigma = self._settings['sigma']
        if self._mus is not None:
            self._nb_center, self._data_dim = mus.shape
        else:
            self._mus = self.randunif(-1, 1, (self._data_dim, ))

    def _sample_data_label_weight(self):
        id_center = self.randint(0, self._nb_center - 1)
        y = self._mus[id_center, :]
        y = y.reshape((self._data_dim,))
        x = self.randnorm(y, sigma=self._sigma)
        return (x, y, 1.0)

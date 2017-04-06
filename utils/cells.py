import random
import logging
import xlearn.utils.general as utg
from ..utils.general import with_config


class Cell(object):
    @with_config
    def __init__(self,
                 cells_in=None,
                 name='Cell',
                 **kwargs):
        if cells_in is None:
            self._cins = []
        else:
            self._cins = cells_in
        self._name = name

    def add_in(self, c, channel=None):
        if channel is None:
            channel = len(self._cins)
        while self.nb_in <= channel:
            self._cins.append(None)
        self._cins[channel] = c

    def _process(self, pumped=None, inputs=None):
        output = []
        if pumped is not None:
            output.append(pumped)
        elif inputs is not None:
            output += [inputs]
        return output

    def _pump(self):
        return None

    def __next__(self):
        pumped = self._pump()
        inputs = []
        if self.nb_in > 0:
            for c in range(self._cins):
                inputs.append(next(c))
        return self._process(pumped, inputs)

    def __iter__(self):
        return self

    @property
    def name(self):
        return self._name

    @property
    def nb_in(self):
        return len(self._cins)


class Counter(Cell):
    """ A conter start from 0.
    Add one each time next(self) if called.
    If counter reaches nb_max, StopIteration will raised.
    """
    @with_config
    def __init__(self,
                 nb_max=None,
                 name='Counter',
                 **kwargs):
        """
        *Pump cell*
        Args:
            name:   *str*,  _Counter_,  name of cell.
            nb_max: *int*,  _-1_,   maxinum state.
        """
        Cell.__init__(self, name=name, **kwargs)
        self._nb_max = nb_max
        self._state = 0

    def _pump(self):
        output = self._state
        if self._state == self.nb_max:
            raise StopIteration
        self._state += 1
        return output

    @property
    def nb_max(self):
        return self._nb_max

    @property
    def state(self):
        return self._state

    def reset(self, value=0):
        self._state = value


class Sampler(Cell):
    # TODO: Add feature of random seed.

    def __init__(self,
                 datas=[],
                 is_shuffle=False,
                 seed=0,
                 name='Sampler',
                 **kwargs):
        """
        Args:
            datas:      list, where data sample from.
            is_shuffle: bool, whether use shuffle.
        """
        Cell.__init__(self, name=name, **kwargs)
        self._datas = datas
        self._nb_datas = len(self._datas)
        if self._nb_datas == 0:
            raise TypeError("Can not construct sampler on empty datas.")
        self._is_shuffle = is_shuffle
        self._seed = seed
        self._counter = Counter(nb_max=self._nb_datas,
                                name=self._name + '/nb_sampled')
        self._ids = list(range(self._nb_datas))
        self._nb_epoch = Counter(name=self._name + '/nb_epoch')
        self.end_of_epoch()

    def end_of_epoch(self):
        if self._is_shuffle:
            random.shuffle(self._ids)
        self._counter.reset()
        next(self._nb_epoch)

    def _pump(self):
        try:
            id_ = next(self._counter)
        except StopIteration:
            self.end_of_epoch()
            id_ = next(self._counter)
        return self._datas[self._ids[id_[0]]]

    @property
    def nb_epoch(self):
        return self._nb_epoch.state

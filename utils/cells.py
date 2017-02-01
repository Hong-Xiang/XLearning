from xlearn.utils.general import with_file_config

class Cell(object):

    def __init__(self, cells_in=None, config_file_names=None, name='Cell', **kwargs):
        if cells_in is None:
            self._cins = []
        else:
            self._cins = cells_in
        self._name = name
        self._settings = self._get_config(config_file_names, )

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

    def __init__(self, nb_max=-1, *args, **kwargs):
        super(Counter, self).__init__(*args, **kwargs)
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

"""
General tiny help routines.
"""
import os
import re
import json
import logging
import sys
import numpy as np
from functools import wraps
from itertools import zip_longest
import time
import logging
import time
import datetime
import tensorflow as tf
import click
from inspect import getfullargspec, signature
from functools import wraps
from pdb import set_trace

DEBUG = True

def analysis_device(device):
    p = re.compile('/([a-z]*):([0-9]*)')
    m = p.match(device)
    device_type = m[1]
    device_id = int(m[2])
    return device_type, device_id


def pp_json(title, cfg_dict, length=30):
    click.echo(title)
    click.echo("=" * length)
    click.echo(json.dumps(cfg_dict, indent=4,
                          separators=[',', '： '], sort_keys=True))
    click.echo("=" * length)


def get_args(func, all_vars):
    d = {}
    sig = signature(func)
    for param in sig.parameters.values():
        if param.kind == param.VAR_POSITIONAL or param.kind == param.VAR_KEYWORD:
            continue
        d.update({param.name: all_vars[param.name]})
    return d


def print_pretty_args(func, all_vars):
    if isinstance(func, click.core.Command):
        func = func.callback
    sig = signature(func)
    d = {}
    for param in sig.parameters.values():
        if param.kind == param.VAR_POSITIONAL or param.kind == param.VAR_KEYWORD:
            continue
        d.update({param.name: all_vars[param.name]})
    prefix = "=" * 30 + "\n"
    prefix += str(func.__name__) + " args:" + "\n"
    prefix += "." * 30 + "\n"
    sets = json.dumps(d, indent=4, separators=[',', ': '], sort_keys=True)
    suffix = "\n" + "=" * 30
    click.echo(prefix + sets + suffix)


def setting_with_priority(settings_list):
    for setting in settings_list:
        if setting is not None:
            return setting
    return None


def config_from_dicts(key, dicts, mode='first'):
    value = None
    for d in dicts:
        tmp = d.get(key)
        if mode == 'first':
            value = tmp
            if value is not None:
                break
        elif mode == 'append':
            if value is None:
                if isinstance(tmp, (list, tuple)):
                    value = list(tmp)
                else:
                    value = [tmp]
            else:
                if isinstance(tmp, (list, tuple)):
                    value += list(tmp)
                else:
                    value.append(tmp)
    return value


def print_global_vars():
    for v in tf.global_variables():
        print(v.name)


class Sentinel:
    pass


def zip_equal(*iterables):
    sen = Sentinel()
    for combo in zip_longest(*iterables, fillvalue=sen):
        for ele in combo:
            if isinstance(ele, Sentinel):
                raise ValueError('Iterables have different length.')
        yield combo


def extend_list(list_input, nb_target):
    if len(list_input) == nb_target:
        return list_input
    if len(list_input) == 1 and nb_target > 1:
        return list_input * nb_target
    else:
        raise ValueError("Can't extend list with len != 1")


def empty_list(length, is_lol=False):
    output = []
    for i in range(length):
        if is_lol:
            output.append([])
        else:
            output.append(None)
    return output


class ExceptionHook:
    instance = None

    def __call__(self, *args, **kwargs):
        if self.instance is None:
            from IPython.core import ultratb
            self.instance = ultratb.FormattedTB(mode='Plain',
                                                color_scheme='Linux', call_pdb=1)
        return self.instance(*args, **kwargs)


def enter_debug():
    sys.excepthook = ExceptionHook()

def add_breakpoint():
    if DEBUG:
        set_trace()

def show_debug_logs():
    """ print debug logging info """
    logger = logging.getLogger()
    sh = logging.StreamHandler(sys.stderr)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG)


def unpack_list_nd(input_, item_type=None, keep_types=(str, np.ndarray)):
    """unpack list of multi dimension into one dimension.
    :param input_: a container type object, will be expand if hasattr '__iter__'
    :param item_type: if not None, none item_type elements will be droped
    :param keep_types: do not expand these types.
    :returns: a list of elements
    """
    result = []
    for list_maybe in input_:
        is_list = hasattr(list_maybe, '__iter__')
        for keep in keep_types:
            if isinstance(list_maybe, keep):
                is_list = False
        if is_list:
            result.extend(unpack_list_nd(list_maybe, item_type))
        else:
            if item_type is None or isinstance(list_maybe, item_type):
                result.append(list_maybe)
    return result


def errmsg(got, required, msg=""):
    """Warp for error message.
    """
    return msg + "got: {0}, required: {1}.".format(got, required)


def seperate_file_name(file):
    """Analysis standard file name.
    """
    pat = re.compile(r'(\D*)(\d+)\.{0,1}(\w*)')
    match = pat.match(file)
    if not match:
        return None, None, None
    prefix = match.group(1)
    id_ = int(match.group(2))
    suffix = match.group(3)
    return prefix, id_, suffix


def form_file_name(prefix, id_, suffix=None):
    """Standard file name of datas for xlearn, given prefix and suffix.
    :param prefix: file name prefix
    :param id_: case if of data
    :param suffix: suffix of data, default = None
    :returns: formed filename
    """
    if suffix == '' or suffix is None:
        filename = prefix + '%09d' % id_
    else:
        filename = prefix + '%09d' % id_ + '.' + suffix
    return filename


def label_name(data_name, case_digit=None, label_prefix=None):
    """Get filename of label to a data name.
    :param data_name:   data filename
    :param case_digit:  number of digits for case. default = None
                        e.g. dataSSSDDDDDD.xxx will share label000DDDDDD.xxx as
                        label file with case_digit = 6.
    :param label_prefix:prefix of label filename. default is same prefix as
                        data file.
    :returns:            label filename
    """
    prefix, id_, suffix = seperate_file_name(data_name)
    if case_digit is not None:
        id_ = str(id_)
        id_ = id_[:-case_digit]
        id_ = int(id_)
    if label_prefix is None:
        label_prefix = prefix
    output = form_file_name(label_prefix, id, suffix)
    return output

# def valid_filename_form():
#     output = r"\w+\d[16].\w"
#     return


def with_config_old(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        fns = kwargs.pop('filenames', None)
        sets = merge_settings(settings=kwargs.pop('settings', None), filenames=fns,
                              default_settings=kwargs.pop('default_settings', None), **kwargs)
        logging.getLogger(__name__).debug(
            "After merge_settings, settings:" + str(sets) + "\nkwargs:" + str(kwargs))
        logging.getLogger(__name__).debug(
            "args:" + str(args))
        kwargs['filenames'] = fns
        return func(*args,  settings=sets, **kwargs)
    return wrapper


def with_config(func):
    """ add support to read keywargs from .json file
    Add a preserved keyword 'filenames' for .json files.
    """
    @wraps(func)
    def wrapper(*args, filenames=None, **kwargs):
        if isinstance(filenames, str):
            filenames = [filenames]
        if filenames is None:
            filenames = []
        paras = getfullargspec(func)
        json_dicts = []
        for fn in filenames:
            with open(fn, 'r') as fin:
                json_dicts.append(json.load(fin))
        if paras.defaults is not None:
            nb_def = len(paras.defaults)
        else:
            nb_def = 0
        def_args = paras.defaults
        if def_args is None:
            def_args = []
        def_keys = paras.args[-nb_def:]
        def_dict = {k: v for k, v in zip(def_keys, def_args)}
        for k in paras.args:
            v = config_from_dicts(k, [kwargs] + json_dicts + [def_dict])
            if v is not None:
                kwargs.update({k: v})
        kwargs.update({'filenames': filenames})
        kwargs.pop('filenames')
        return func(*args, filenames=filenames, **kwargs)
    return wrapper


def merge_settings(settings=None, filenames=None, default_settings=None, **kwargs):
    """Merge settings from multiple file and args into one
    """
    if settings is None:
        settings = {}

    if default_settings is not None:
        settings.update(default_settings)

    if filenames is None:
        filenames = ()
    if not isinstance(filenames, (list, tuple)):
        filenames = (filenames,)
    for filename in filenames:
        filename = os.path.abspath(filename)
        if not os.path.isfile(filename):
            continue
        with open(filename, 'r') as file_conf:
            tmp = json.load(file_conf)
            tmp = dict(filter(lambda x: x[1] is not None, tmp.items()))
            settings.update(tmp)

    filted_kwargs = dict(filter(lambda x: x[1] is not None, kwargs.items()))

    settings.update(filted_kwargs)

    return settings


def check_same_len(shape0, shape1, ext_msg=''):
    if len(shape0) != len(shape1):
        raise ValueError(errmsg(shape0.shape, shape1.shape,
                                ext_msg + "Need to be same length, "))


def filename_filter(filenames, prefix, suffix):
    dirs = []
    files = []
    for filename in filenames:
        path_full = os.path.abspath(filename)
        if os.path.isdir(filename):
            dirs.append(path_full)
        if os.path.isfile(path_full):
            filename_tail = os.path.basename(path_full)
            prefix, id, suffix = seperate_file_name(filename_tail)
            if prefix is None:
                continue
            else:
                files.append(path_full)
    return files, dirs


class ProgressTimer:
    def __init__(self, nb_steps=100, min_elp=1.0):
        self._nb_steps = nb_steps
        self._step = 0
        self._start = None
        self._elaps = None
        self._pre = None
        self._min_elp = min_elp
        self.reset()

    def reset(self):
        self._start = time.time()
        self._pre = 0.0
        self._step = 0

    def event(self, step=None, msg='None'):
        if step is None:
            step = self._step
            self._step += 1
        else:
            self._step = step

        self._elaps = time.time() - self._start
        if self._elaps - self._pre < self._min_elp:
            return
        comp_percen = float(step) / float(self._nb_steps)
        if comp_percen > 0:
            eta = (1 - comp_percen) * self._elaps / comp_percen
        else:
            eta = None

        time_pas = str(datetime.timedelta(seconds=int(self._elaps)))
        time_int_v = self._elaps / (step + 1.0)
        if time_int_v < 60:
            time_int_msg = '%0.2fs/it' % (time_int_v)
        else:
            time_int_msg = str(datetime.timedelta(
                seconds=int(time_int_v)))
        if eta is None:
            time_eta = 'UKN'
        else:
            time_eta = str(datetime.timedelta(seconds=int(eta)))
        click.echo("i=%6d, %s, [%s<%s] :" %
                   (step, time_int_msg, time_pas, time_eta) + msg)
        self._pre = self._elaps

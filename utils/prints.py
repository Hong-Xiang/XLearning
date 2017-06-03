""" some utilities for prints """
import json
from inspect import getfullargspec, signature
from functools import wraps
import click
import tensorflow as tf

def pprint(contents, file=None):
    click.echo(contents, file=file)


def hline(level=0, file=None):
    if level == 0:
        pprint("=" * 60, file)
    elif level == 1:
        pprint("-" * 60, file)


class SpecialEncoder(json.JSONEncoder):    
    SPECIAL_TYPES = (tf.Variable, tf.Tensor, tf.Operation)
    def default(self, obj):
        if isinstance(obj, SpecialEncoder.SPECIAL_TYPES):
            return obj.__str__()
        return json.JSONEncoder.default(self, obj)

def pp_json(dict_to_print, title=None, length=30, file=None):
    hline(file)
    if title is not None:
        pprint(title, file)
        hline(1, file)
    pprint(json.dumps(dict_to_print, indent=4,
                      separators=[',', '： '], sort_keys=True, cls=SpecialEncoder), file)
    hline(file)


def get_args(func, all_vars):
    d = {}
    sig = signature(func)
    for param in sig.parameters.values():
        if param.kind == param.VAR_POSITIONAL or param.kind == param.VAR_KEYWORD:
            continue
        d.update({param.name: all_vars[param.name]})
    return d


def pp_args(func, all_vars):
    """ pretty print of args of a function """
    if isinstance(func, click.core.Command):
        func = func.callback
    sig = signature(func)
    d = {}
    for param in sig.parameters.values():
        if param.kind == param.VAR_POSITIONAL or param.kind == param.VAR_KEYWORD:
            continue
        d.update({param.name: all_vars[param.name]})
    hline()
    pp_json(d, str(func.__name__))
    hline()

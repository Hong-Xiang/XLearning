""" some utilities for prints """
import json
from inspect import getfullargspec, signature
from functools import wraps
import click


def pprint(contents):
    click.echo(contents)


def hline(level=0):
    if level == 0:
        pprint("=" * 60)
    elif level == 1:
        pprint("-" * 60)


def pp_json(dict_to_print, title=None, length=30):
    hline()
    if title is not None:
        pprint(title)
        hline(1)
    pprint(json.dumps(dict_to_print, indent=4,
                      separators=[',', '： '], sort_keys=True))
    hline()


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

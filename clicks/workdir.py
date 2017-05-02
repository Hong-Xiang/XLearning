""" click app: workdir """
import sys
import os
import re
import shutil
from pathlib import Path
import json
from inspect import getfullargspec, signature
import click
from xlearn.utils.prints import pp_json, pprint


@click.group()
def workdir():
    """ Work directory utils. """
    click.echo("WORKDIR UTILS CALLED.")


@workdir.command()
@click.option('--target', '-t', type=str, default='.')
@click.option('--kind', '-k', type=str)
@click.option('--name', '-n', type=str)
@click.option('--filenames', '-fn', multiple=True, type=str)
def cfgs(target, kind=None, name=None, **kwargs):
    """ copy config files (automatically adding .json suffix)."""
    click.echo("COPY CONFIGS...")
    if not name.endswith('.json'):
        name += '.json'
    target = Path(target)
    cfg_file = Path(os.environ['PATH_XLEARN']) / 'configs' / kind / name
    cfgs_dict = {
        'target': str(target),
        'kind': kind,
        'name': name,
        'origin': str(cfg_file)
    }
    pp_json(cfgs_dict, 'COPY CONFIG OPTIONS:')
    shutil.copy(cfg_file, name)


@workdir.command()
@click.option('--target', '-t', type=str, default='.')
@click.option('--is_recurrent', '-r', is_flag=True)
@click.option('--level', '-l', type=int, help="Level of clean.", default=0)
@click.option('--no_action', '-n', is_flag=True)
@click.option('--keep', '-k', type=int, multiple=True, help='step of save ckpt to keep.')
def clean(target='.', level=0, is_recurrent=False, no_action=False, keep=None):
    clean_core(target, level, is_recurrent, no_action, keep)


def clean_core(target='.', level=0, is_recurrent=False, no_action=False, keep=None):
    """ Clean work directory. Level 0: clean *.err and redudent checkpoints. Level 1: All out, err, save, log etc."""
    p = Path(target)
    pprint("Clean:\t" + str(p.absolute()))
    pprint("LEVEL: %d" % level)
    if keep is None:
        keep = []
    pprint('TO KEEP:')
    pprint(keep)

    save_re = r'save-([0-9]+).*'
    prog = re.compile(save_re)
    max_step = -1
    files = list(p.iterdir())
    for f in files:
        m = prog.match(f.name)
        if m:
            step = int(m.group(1))
            if step > max_step:
                max_step = step
    pprint("Found max_step: %d" % max_step)
    for f in files:
        m = prog.match(f.name)
        if m:
            step = int(m.group(1))
            if step < max_step or level > 0:
                if step in keep:
                    continue
                if no_action:
                    print('TO REMOVE: ', f.absolute())
                else:
                    os.remove(f.absolute())
        pout = re.compile(r'[0-9]+\.out')
        if pout.match(f.name) and level > 0:
            if no_action:
                print('TO REMOVE: ', f.absolute())
            else:
                os.remove(os.path.abspath(f))
        perr = re.compile(r'[0-9]+\.err')
        if perr.match(f.name):
            if no_action:
                print('TO REMOVE: ', f.absolute())
            else:
                os.remove(f.absolute())
        if f.name == 'checkpoint' and level > 0:
            if no_action:
                print('TO REMOVE: ', os.path.abspath(f.absolute()))
            else:
                os.remove(f.absolute())
        if f.name == 'log' and level > 0:
            if no_action:
                print('TO REMOVE: ', os.path.abspath(f.absolute()))
            else:
                shutil.rmtree(f.absolute())
    if is_recurrent:
        for cp in p.iterdir():
            if cp.is_dir():
                clean_core(cp, level, is_recurrent, no_action, keep)


if __name__ == "__main__":
    workdir()

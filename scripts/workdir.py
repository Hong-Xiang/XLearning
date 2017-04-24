import sys
import os
import click
import re
import shutil
from pathlib import Path
from xlearn.utils.general import print_pretty_args

@click.group()
def workdir():
    """ Work directory utils. """
    click.echo("WORKDIR UTILS:")

@workdir.command()
@click.option('--kind', '-k', type=str)
@click.option('--name', '-n', type=str)
@click.option('--filenames', '-fn', multiple=True, type=str)
def cfgs(kind, name, **kwargs):
    """ copy config files """
    print_pretty_args(cfgs, locals())
    name += '.json'
    cfg_file = Path(os.environ['PATH_XLEARN']) / 'configs' / kind / name
    shutil.copy(cfg_file, name)

@workdir.command()
@click.option('--no_save', is_flag=True, help="Remove all save files. Default will keep the last one.")
@click.option('--no_out', is_flag=True, help="Remove *.out file. Default will only remove *.err files.")
@click.option('--no_action', is_flag=True)
@click.option('--keep', '-k', type=int, multiple=True)
def clean(no_save, no_out=True, no_action=False, keep=None):
    """ Clean work directory """
    click.echo("Clean work directory.")
    files = os.listdir('.')
    save_re = r'save-([0-9]+).*'
    prog = re.compile(save_re)
    max_step = -1
    for f in files:
        m = prog.match(f)
        if m:
            step = int(m.group(1))
            if step > max_step:
                max_step = step
    if keep is None:
        keep = []
    for f in files:
        m = prog.match(f)
        if m:
            step = int(m.group(1))
            if step < max_step:
                if step in keep:
                    continue
                if no_action:
                    print('TO REMOVE: ', os.path.abspath(f))
                else:
                    os.remove(os.path.abspath(f))
        pout = re.compile(r'[0-9]+\.out')
        if pout.match(f) and no_out:
            if no_action:
                print('TO REMOVE: ', os.path.abspath(f))
            else:
                os.remove(os.path.abspath(f))
        perr = re.compile(r'[0-9]+\.err')
        if perr.match(f):
            if no_action:
                print('TO REMOVE: ', os.path.abspath(f))
            else:
                os.remove(os.path.abspath(f))

if __name__ == "__main__":
    workdir()
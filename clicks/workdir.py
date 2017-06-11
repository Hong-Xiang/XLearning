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
from xlearn.utils.dataset import analysis_dataset

@click.group()
def workdir():
    """ Work directory utils. """
    click.echo("WORKDIR UTILS CALLED.")

@workdir.command()
@click.option('--filename', '-f', type=str)
def analysis(filename):
    click.echo("ANALYSIS OF: {}.".format(filename))
    if 'h5' in filename:
        infos = analysis_dataset(filename)
        # pp_json(infos, "PARAS OF DATASET:")
        click.echo(infos)
        

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
    shutil.copy(cfg_file, str(kind)+'.'+name)

@workdir.command()
@click.option('--target', '-t', type=str, default='.')
@click.option('--net', '-n', type=str)
@click.option('--dataset', '-d', type=str)
@click.option('--task', '-a', type=str, default='train')
def new(target, net, dataset, task, **kwargs):
    """ copy config files (automatically adding .json suffix)."""
    click.echo("COPY CONFIGS...")
    if not net.endswith('.json'):
        net += '.json'
    if not dataset.endswith('.json'):
        dataset += '.json'
    if not task.endswith('.json'):
        task += '.json'
    target = Path(target)
    net_file = Path(os.environ['PATH_XLEARN']) / 'configs' / 'net' / net
    data_file = Path(os.environ['PATH_XLEARN']) / 'configs' / 'dataset' / dataset
    task_file = Path(os.environ['PATH_XLEARN']) / 'configs' / 'train' / task
    cfgs_dict = {
        'target': str(target),
        'net': net,
        'net_origin': str(net_file),
        'dataset': dataset,
        'data_origin': str(data_file),
        'task': task,
        'task_origin': str(task_file)
    }
    clean_core(str(target.absolute()), level=100)
    pp_json(cfgs_dict, 'COPY CONFIG OPTIONS:')
    shutil.copy(net_file, 'net.'+net)
    shutil.copy(data_file, 'data.'+dataset)
    shutil.copy(task_file, 'task.'+task)
    

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

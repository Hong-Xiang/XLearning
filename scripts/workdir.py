import sys
import os
import click
import re

@click.group()
def workdir():
    """ Work directory utils. """
    click.echo("WORKDIR UTILS:")

@workdir.command()
@click.option('--no_save', is_flag=True, help="Remove all save files. Default will keep the last one.")
@click.option('--no_out', is_flag=True, help="Remove *.out file. Default will only remove *.err files.")
@click.option('--no_action', is_flag=True)
def clean(no_save, no_out=True, no_action=False):
    """ Clean work directory """
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
    for f in files:
        m = prog.match(f)
        if m:
            step = int(m.group(1))
            if step < max_step:
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
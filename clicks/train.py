import click
import json
import os
from xlearn.utils.general import enter_debug
from xlearn.run.exp import run


@click.group()
def train():
    """ Net train utils. """
    click.echo("NET TRAIN UTILS CALLED.")

@train.command()
def auto():
    run()
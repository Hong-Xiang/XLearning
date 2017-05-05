import click
from xlearn.utils.general import enter_debug
from xlearn.clicks.workdir import workdir

@click.group()
@click.option('--cfg', default='config.json')
@click.option('--debug', is_flag=True, default=False)
def xln(cfg, debug):
    if debug:
        print("ENTER DEBUG MODE")
        enter_debug()

xln.add_command(workdir)


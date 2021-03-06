import click
from xlearn.utils.general import enter_debug
from xlearn.clicks.workdir import workdir
from xlearn.clicks.train import train

@click.group()
@click.option('--cfg', default='config.json')
@click.option('--debug', is_flag=True, default=False)
def xln(cfg, debug):
    if debug:
        print("ENTER DEBUG MODE")
        enter_debug()

xln.add_command(workdir)
xln.add_command(train)

if __name__ == "__main__":
    xln()


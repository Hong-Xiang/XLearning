import click
import json
import os
import sys
from pathlib import Path
import numpy as np
from xlearn.utils.prints import pprint, pp_json
from xlearn.datasets.sinogram import Sinograms
from xlearn.net_tf import SRSino8v5


@click.group()
def gen_sr():
    pprint("GEN SR UTILS")


@gen_sr.command()
@click.command('--target', '-t', type=str, default='.')
@click.command('--level', '-l', type=int)
def gendata(target='.', level=3):
    with Sinograms(batch_size=1, mode='test', label_down_sample=0, data_down_sample=3, crop_shape=[363, 720, 1]) as dataset:
        ss = next(dataset)
        print("low resolution shape,", ss[0].shape)
        print("label left shape,", ss[1][0].shape)
        print("label right shape,", ss[1][1].shape)
        print("label full shape,", ss[1][2].shape)
        np.save(Path(target) / 'ss0.npy', ss[0])
        np.save(Path(target) / 'ss10.npy', ss[1][0])
        np.save(Path(target) / 'ss11.npy', ss[1][1])
        np.save(Path(target) / 'ss12.npy', ss[1][2])


@gen_sr.command()
@click.option('--max_down', '-m', type=int, default=3)
def build(max_down):
    pgen = Path(os.environ.get('PATH_XLEARN'))
    pgen = pgen / 'clicks' / 'gen_sr.py'
    pgen = pgen.absolute()
    pgen = str(pgen)
    with open(Path('run_gen.sh'), 'w') as fin:
        print('python ' + pgen + ' gendata -l ' +
              str(max_down) + ' -t inputs', file=fin)
        if max_down > 2:
            print('cp inputs/ss0.npy ip.npy')
            print('cd net8x', file=fin)
            print('python ' + pgen + ' run -l 3', file=fin)
            print('cd ..', file=fin)
            print('cp net8x/infer8x.npy input')
    os.system("chmod +x run_gen.sh")


@gen_sr.command()
@click.command('--level', '-l', type=int, default=3)
@click.command('--filenames', '-fn', type=str, multiple=True)
@click.command('--ipt', '-i', type=str)
@click.command('--opt', '-o', type=str)
@click.command('--period', '-p', type=int)
def run(level=3, filenames=None, ipt=None, opt=None, period=None):
    cfgs = {
        'level': level,
        'filenames': filenames,
        'ip': ipt,
        'out': opt,
    }
    pp_json(cfgs, "RUN WITH CONFIG")
    # check target folder:
    ipt = np.load(ipt)
    net = SRSino8v5(filenames=filenames)
    out = net.predict_fullsize(ipt, period)
    np.save(opt, out)


@gen_sr.command()
@click.option('--net8x', type=str, default='net8x')
@click.option('--net4x', type=str, default='net4x')
@click.option('--net2x', type=str, default='net2x')
@click.option('--target', type=str, default='target')
@click.option('--max_down', type=int, default=3)
def prepare(net_8x, net_4x, net_2x, target, max_down):

    if max_down >= 3:
        pass


if __name__ == "__main__":
    gen_sr()

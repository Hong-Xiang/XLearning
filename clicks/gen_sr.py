import click
import json
import os
import sys
from pathlib import Path
import numpy as np
from xlearn.utils.prints import pprint
from xlearn.datasets.sinogram2 import Sinograms2

@click.group()
def gen_sr():
    pprint("GEN SR UTILS")

@gen_sr.command()
@click.command('--level', '-l', type=int)
def gendata(level):
    with Sinograms2(batch_size=1, mode='test', label_down_sample=1, data_down_sample=3, crop_shape=[363, 720, 1]) as dataset:
    ss = next(dataset)
    print(ss[0].shape)
    print(ss[1][0].shape)
    print(ss[1][1].shape)
    print(ss[1][2].shape)
    np.save('/home/hongxwing/Workspace/srsino/w20170428/recon/ss0.npy', ss[0])
    np.save('/home/hongxwing/Workspace/srsino/w20170428/recon/ss10.npy', ss[1][0])
    np.save('/home/hongxwing/Workspace/srsino/w20170428/recon/ss11.npy', ss[1][1])
    np.save('/home/hongxwing/Workspace/srsino/w20170428/recon/ss12.npy', ss[1][2])
    plt.imshow(np.float32(ss[1][2][0,:,:,0]))
    plt.figure()
    plt.imshow(np.float32(ss[0][0,:,:,0]))

@gen_sr.command()
@click.option('--max_down', '-m', type=int, default=3)
def build(max_down):
    pgen = Path(os.environ.get('PATH_XLEARN'))
    pgen = pgen / 'clicks' / 'gen_sr.py'
    pgen = pgen.absolute()
    pgen = str(pgen)
    with open(Path('run_gen.sh'), 'w') as fin:
        print('python ' + pgen + ' gendata -l ' + str(max_down), file=fin)
        for ld in range(max_down, 0, -1):
            print('python ' + pgen + ' run -l ' + str(ld), file=fin)
    os.system("chmod +x run_gen.sh")

@gen_sr.command()
@click.command('--level', '-l', type=int)
def run(level):
    if level is None:
        raise TypeError('run level not specified!')
    #check target folder:
    p = Path('.') / 'recon'
    if p.is_dir() is False:
        raise 

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

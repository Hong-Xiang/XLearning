import click
import json
import time

import xlearn.datasets as datasets
import xlearn.net_tf as nets2
from xlearn.utils.general import pp_json, ProgressTimer


@click.group()
def train():
    click.echo("TRAIN ROUTINES:")


@train.command()
@click.option('--cfg', '-c', type=str)
def train2(cfg='train.json',
           **kwargs):
    with open(cfg, 'r') as fin:
        cfgs = json.load(fin)
        pp_json('TRAIN 2 CONFIGS:', cfgs)
    dataset_class = getattr(datasets, cfgs['dataset'])
    net_class = getattr(nets2, cfgs['net'])
    filenames = cfgs.get('filenames')
    load_step = cfgs.get('load_step')
    net = net_class(filenames=filenames, **kwargs)
    net.build()
    if load_step is not None:
        net.load(load_step=load_step)

    pre_sum = time.time()
    pre_save = time.time()
    lrs = cfgs['lrs']
    total_step = cfgs['total_step']
    summary_freq = cfgs['summary_freq']
    save_freq = cfgs['save_freq']
    with dataset_class(filenames=filenames) as dataset_train:
        with dataset_class(filenames=filenames, mode='test') as dataset_test:
            pt = ProgressTimer(total_step * len(lrs))
            cstep = 0
            for lrv in lrs:
                net.learning_rate_value = lrv
                for i in range(total_step):
                    ss = next(dataset_train)
                    loss_v, _ = net.train(ss)
                    pt.event(cstep, msg='loss %e.' % loss_v)
                    cstep += 1
                    now = time.time()
                    if now - pre_sum > summary_freq:
                        ss = next(dataset_train)
                        net.summary(ss, True)
                        ss = next(dataset_test)
                        net.summary(ss, False)
                        pre_sum = now
                    if now - pre_save > save_freq:
                        net.save()
                        pre_save = now
    net.save()

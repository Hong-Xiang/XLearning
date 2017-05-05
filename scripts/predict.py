import tensorflow as tf
import click
import xlearn.datasets as datasets
import xlearn.net_tf as nets2


@click.group()
def predict():
    click.echo('PREDICT ROUTINES CALLED.')


@predict.command()
@click.option('--cfg', '-c', type=str)
@with_config
def predict_sino8v3(cfg='train.json',
                    **kwargs):
    with open(cfg, 'r') as fin:
        cfgs = json.load(fin)
        pp_json('PREDICT CONFIGS:', cfgs)
    dataset_class = getattr(datasets, cfgs['dataset'])
    net_class = getattr(nets2, cfgs['net'])
    filenames = cfgs.get('filenames')
    load_step = cfgs['load_step']
    net = net_class(filenames=filenames, **kwargs)
    net.build()
    net.load(load_step=load_step)
    with dataset_class(filenames=filenames) as dataset:
        for i in range(total_step):
            ss = next(dataset)
            net.predict
            pt.event(i, msg='loss %e.' % loss_v)
            now = time.time()
            if now - pre_sum > 120:
                ss = next(dataset_train)
                net.summary(ss, True)
                ss = next(dataset_test)
                net.summary(ss, False)
                pre_sum = now
            if now - pre_save > 600:
                net.save()
                pre_save = now
        net.save()


@predict.command()
@click.option('--filenames', '-fn', multiple=True, type=str)
@click.option('--load_step', type=int)
@click.option('--total_step', type=int)
@with_config
def train_sino8v3_pet(load_step=None,
                      total_step=None,
                      filenames=[],
                      **kwargs):
    print("TRAINGING v3 net on PET data.")
    net = SRSino8v3(filenames=filenames, **kwargs)
    net.build()
    if load_step is not None:
        net.load(load_step=load_step)

    pre_sum = time.time()
    pre_save = time.time()
    with SinogramsPETRebin(filenames=filenames) as dataset_train:
        with SinogramsPETRebin(filenames=filenames, mode='test') as dataset_test:
            pt = ProgressTimer(total_step)
            for i in range(total_step):
                ss = next(dataset_train)
                loss_v, _ = net.train(ss)
                pt.event(i, msg='loss %e.' % loss_v)
                now = time.time()
                if now - pre_sum > 120:
                    ss = next(dataset_train)
                    net.summary(ss, True)
                    ss = next(dataset_test)
                    net.summary(ss, False)
                    pre_sum = now
                if now - pre_save > 600:
                    net.save()
                    pre_save = now
        net.save()

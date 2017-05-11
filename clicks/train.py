import click
import json
from xlearn.utils.general import enter_debug
from xlearn.utils.prints import pp_json
import xlearn.nets as nets
import xlearn.datasets as datasets


@click.group()
def train():
    """ Net train utils. """
    click.echo("NET TRAIN UTILS CALLED.")


@train.command()
def auto():
    with open('task.train.json', 'r') as fin:
        train_task = json.load(fin)
        pp_json(train_task, "TASK PARAMS")
        net_name = train_task['net_name']
        dataset_name = train_task['dataset_name']
        steps = train_task['steps']
        decay = train_task['decay']
        load_step = train_task.get('load_step')
        filenames = train_task.get('filenames', [])
    train_core(net_name, dataset_name, filenames, steps, decay, load_step)

def train_core(net_name, dataset_name, filenames, steps, decay, load_step):        
        data_cls = getattr(datasets, dataset_name)
        net_cls = getattr(nets, net_name)
        net = net_cls(filenames=filenames, load_step=load_step)
        net.init()
        with data_cls(filenames=filenames, mode='train') as dataset_train:
            with data_cls(filenames=filenames, mode='test') as dataset_test:
                net.set_dataset({'train': dataset_train, 'test': dataset_test})
                net.train(steps=steps, decay=decay)
                net.save()

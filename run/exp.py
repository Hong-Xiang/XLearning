import tensorflow as tf
import json
import xlearn.datasets as datasets
import xlearn.nets as nets
from xlearn.utils.prints import pp_json

def run(task_file=None):
    if task_file is None:
        task_file = 'task.train.json'
    with open(task_file, 'r') as fin:
        task = json.load(fin)
        pp_json(task, "TASK PARAMS")
        net_name = task['net_name']
        dataset_name = task['dataset_name']
        exp_name = task['exp_name']
        load_step = task.get('load_step')
        net_files = task.get('net_file')
        data_files = task.get('data_file')

        if exp_name == 'train':        
            steps = task['steps']
            decay = task['decay']
            sub_task = task['task']                                          
            train_core(net_name, dataset_name, net_files, data_files, steps, decay, load_step, sub_task)
        if exp_name == 'init_net':
            init_test(net_name, net_files)

def train_core(net_name, dataset_name, net_files, data_files, steps, decay, load_step, task=None):        
        data_cls = getattr(datasets, dataset_name)
        net_cls = getattr(nets, net_name)
        net = net_cls(filenames=net_files, load_step=load_step)
        net.init()
        with data_cls(filenames=data_files, mode='train') as dataset_train:
            with data_cls(filenames=data_files, mode='test') as dataset_test:
                net.set_dataset_auto(dataset_train, dataset_test)
                net.train(task, steps=steps, decay=decay)
                net.save()

# def init_test(net_name, dataset_name, net_files, data_files):   

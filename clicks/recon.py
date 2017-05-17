import click
import json
import os
from xlearn.utils.general import enter_debug
from xlearn.utils.prints import pp_json
import xlearn.nets as nets
import xlearn.datasets as datasets
import numpy as np

# @click.group()
# def train():
#     """ Net train utils. """
#     click.echo("NET TRAIN UTILS CALLED.")


# @train.command()
# def auto():
#     with open('task.train.json', 'r') as fin:
#         train_task = json.load(fin)
#         pp_json(train_task, "TASK PARAMS")
#         net_name = train_task['net_name']
#         dataset_name = train_task['dataset_name']
#         steps = train_task['steps']
#         decay = train_task['decay']
#         load_step = train_task.get('load_step')
#         filenames = train_task.get('filenames', [])
#     train_core(net_name, dataset_name, filenames, steps, decay, load_step)

# def train_core(net_name, dataset_name, filenames, steps, decay, load_step):        
#         data_cls = getattr(datasets, dataset_name)
#         net_cls = getattr(nets, net_name)
#         net = net_cls(filenames=filenames, load_step=load_step)
#         net.init()
#         with data_cls(filenames=filenames, mode='train') as dataset_train:
#             with data_cls(filenames=filenames, mode='test') as dataset_test:
#                 net.set_dataset('train', dataset_train)
#                 net.set_dataset('test', dataset_test)
#                 net.train(steps=steps, decay=decay)
#                 net.save()

from tqdm import tqdm

def main():
    enter_debug()
    is_net3 = False
    if is_net3:
        net = nets.SRNet3(filenames=['data.sino8x.json', 'net.srnet1.json'], batch_size=2, low_shape=[90, 90], high_shape=[360, 360], nb_down_sample=2, load_step=-1)
    else:
        net = nets.SRNet4(filenames=['data.sino8x.json', 'net.srnet1.json'], batch_size=2, low_shape=[45, 45], high_shape=[360, 360], nb_down_sample=2, load_step=-1)
    # net = nets.SRNet3(filenames=['data.sino8x.json', 'net.srnet1.json'], load_step=-1)    
    net.init()
    # with datasets.SinoShep(filenames='data.sino8x.json') as dataset:
        # ss = dataset.sample()
    ss = np.load('to_sr.npy').item()
    nb_images = ss['data'].shape[0]
    srs = []
    its = []
    for i in tqdm(range(nb_images//2)):
        if is_net3:
            feed = ss
        else:
            feed = {
            'data3': ss['data3'][2*i:2*i+2, :, :, :],
            'data': ss['data2'][2*i:2*i+2, :, :, :],
            'data2': ss['data2'][2*i:2*i+2, :, :, :],
            'data1': ss['data1'][2*i:2*i+2, :, :, :],
            'data0': ss['data0'][2*i:2*i+2, :, :, :]}        
        pred = net.predict(feed)             
        srs.append(pred['inference'])       
        its.append(pred['interp'])
        print(pred['inference'].shape)
    pred_sr = np.concatenate(srs, axis=0)
    pred_it = np.concatenate(its, axis=0)
    # pred = net.predict(ss)    
    np.save('sr.npy', pred_sr)
    np.save('it.npy', pred_it)
    # np.savez('sr_res.npz', sr=pred_sr, it=pred_it)


if __name__ == "__main__":
    main()

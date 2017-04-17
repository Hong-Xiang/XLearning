import os
import json
import shutil

def sbatch_all(arch='k80'):
    dirs = os.listdir('.')
    paths = [os.path.abspath(d) for d in dirs]
    for p in paths:
        if os.path.isdir(p):
            os.system('cd ' + p + '; chmod +x work.sh; sbatch' + arch + '.slurm')

def prepare_jsons(def_c, replica=1, batch_size=None, input_shape=None, lr=None, filters=None, depths=None, blocks=None, cores=None, is_bn=None):
    net_c = dict(def_c)
    ds_c2x = {
        "batch_size": batch_size,
        "crop_shape": [input_shape[0], input_shape[1]*2],
        "data_down_sample": 1,
        "label_down_sample": 0
    }
    ds_c4x = {
        "batch_size": batch_size,
        "crop_shape": [input_shape[0], input_shape[1]*4],
        "data_down_sample": 2,
        "label_down_sample": 1
    }
    ds_c8x = {
        "batch_size": batch_size,
        "crop_shape": [input_shape[0], input_shape[1]*8],
        "data_down_sample": 3,
        "label_down_sample": 2
    }
    
    if batch_size is not None:
        net_c['batch_size'] = batch_size
    if input_shape is not None:
        net_c['input_shape'] = input_shape
    if lr is not None:
        net_c['lrs'] = lr
    if filters is not None:        
        net_c['filters'] = filters
    if depths is not None:
        net_c['depths'] = depths
    if blocks is not None:
        net_c['blocks'] = blocks
    if cores is not None:
        net_c['cores'] = cores
    if is_bn is not None:
        net_c['is_bn'] = is_bn
    for i in range(replica):
        path = "rs%df%dd%db%dc%dl%0.1e"%(net_c['batch_size'], net_c['filters'], net_c['depths'], net_c['blocks'], net_c['cores'], net_c['lrs'])        
        if net_c['is_bn']:
            path += 'bn'
        path = os.path.abspath(path)
        os.mkdir(path)
        with open(os.path.join(path, 'sino2_shep8x.json'), 'w') as fout:
            json.dump(ds_c8x, fout, indent=4, separators=[',', ': '], sort_keys=True)
        with open(os.path.join(path, 'sino2_shep4x.json'), 'w') as fout:
            json.dump(ds_c4x, fout, indent=4, separators=[',', ': '], sort_keys=True)
        with open(os.path.join(path, 'sino2_shep2x.json'), 'w') as fout:
            json.dump(ds_c2x, fout, indent=4, separators=[',', ': '], sort_keys=True)
        with open(os.path.join(path, 'srsino8v2.json'), 'w') as fout:
            json.dump(net_c, fout, indent=4, separators=[',', ': '], sort_keys=True)
        xlearn_path = os.environ['PATH_XLEARN']
        with open(os.path.join(path, 'work.sh'), 'w') as fout:
            print("python " + xlearn_path+r"/scripts/main.py train_sino8v2 --total_step 1000 --step8 4 --step4 2 --step2 1 --sumf 1 --savef 20", file=fout)
        shutil.copy(xlearn_path+r"/template/k80.slurm", os.path.join(path, 'k80.slurm'))
        shutil.copy(xlearn_path+r"/template/p100.slurm", os.path.join(path, 'p100.slurm'))

def grid_search_sino8():
    with open('gsc.json', 'r') as fin:
        gsc = json.load(fin)
    with open('srsino8v2.json', 'r') as fin:
        def_nc = json.load(fin)    
    batch_size = gsc['batch_size']
    input_shape = gsc['input_shape']
    
    for lr in gsc['lr']:
        for filters in gsc['filters']:
            for depths in gsc['depths']:
                for blocks in gsc['blocks']:
                    prepare_jsons(def_nc, replica=1, batch_size=batch_size, input_shape=input_shape, lr=lr, filters=filters, depths=depths, blocks=blocks, cores=3, is_bn=True)
                    

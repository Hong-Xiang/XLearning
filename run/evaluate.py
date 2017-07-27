# Scripts to evaluate SR networks.
import json
from pprint import pprint
from xlearn.nets import SRMSL
from xlearn.utils.io import load_jsons
from xlearn.utils.recon import pad_period
from tqdm import tqdm, tqdm_notebook

def _get_net(config_files):
    cfg = load_jsons(config_files)
    pprint(cfg)
    nb_down_sample = cfg['nb_down_sample']
    low_shape=[full_shape[0]//(2**nb_down_sample), full_shape[1]*2//(2**nb_down_sample)]
    net = SRMSL(filenames=config_files, low_shape=low_shape, batch_size=2, load_step=-1)
    net.init()
    return net

def _generate_interps(sino_data, net):
    print("Generating interpolated sinograms.")
    ss_itp = dict(sino_data)
    ss_itp['keep_prob'] = 1.0
    itpss = []
    batch_net = net.p.batch_size
    for i in tqdm(range(ss['data'].shape[0]//batch_net)):
        ss_batch = dict()
        for k in ss_itp:            
            if isinstance(ss_itp[k], np.ndarray) and ss_itp[k].ndim > 2:
                ss_batch[k] = ss_itp[k][i*batch_net:(i+1)*batch_net]
            else:
                ss_batch[k] = ss_itp[k]
        itps = net.run('interp', ss_batch)
        itpss.append(itps['itp'])
    itpss = np.concatenate(itpss, axis=0)
    return itpss

def _generate_super_resolutions(sino_data, net):
    res = net.predict_auto(sino_data, sub_task='main', batch_size=net.p.batch_size)
    return res['inf']

def _padding_sinos(sinos_dict, net):
    pass

def generate_sinos(sino_data, config_files, full_shape):
    print("Start evluating network: generating sinograms")
    net = _get_net(config_files)
    sinos_itp = _generate_interps(sino_data, net)
